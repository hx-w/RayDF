# -*- coding: utf-8 -*-

import os
import numpy as np
import utils
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import yaml
import torch
import cv2
from tqdm import tqdm
import time
import seaborn as sns

from networks.RayDFNet import RayDistanceField
from render.shade import shading_phong


class ImplicitDrawable:
    def __init__(
        self, embedding=None, 
        Mscale: np.array = [1., 1., 1.],
        Mrot: np.array = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        Mtrans: np.array = [0., 0., 0.],
        raw_color: np.array = [100., 100., 100.],
        is_sim: bool = False
    ):
        self.latent = embedding
        self.Mscale = np.array(Mscale)
        self.Mrot   = np.array(Mrot)
        self.Mtrans = np.array(Mtrans)
        self.raw_color = np.array(raw_color, dtype=np.uint8)
        self.is_sim = is_sim

class PLight:
    def __init__(self, pos: np.array, color: np.array):
        self.pos   = np.array(pos)
        self.color = np.array(color)

class GBuffer:
    def __init__(self, w, h):
        self.depth_buffer  = np.ones(shape=(w, h)) * np.inf
        self.normal_buffer = np.ones(shape=(w, h, 3)) * np.array([0., 0., 1.])
        
        self.raw_color = np.ones(shape=(w, h, 3), dtype=np.uint8) * np.array([100, 100, 100], dtype=np.uint8) # 255

class ImplicitScene:
    def __init__(self):
        self.drawables: dict = {}  # {'T11_RDF': [ImplicitDrawable]}
        self.models: dict    = {}  # {'T11_RDF': NeuralNetworks}
        self.lights = []
        self.lights_mat = None
        self.gbuffer = None
        
    def add_drawable(
        self,
        model_name: str,
        model,
        embedding=None,
        Mscale: np.array = [1., 1., 1.],
        Mrot: np.array = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        Mtrans: np.array = [0., 0., 0.],
        raw_color: np.array = [100., 100., 100.],
        is_sim: bool = False
    ):
        if model_name not in self.models.keys():
            self.models[model_name] = model
            self.drawables[model_name] = []
        
        self.drawables[model_name].append(ImplicitDrawable(embedding, Mscale, Mrot, Mtrans, raw_color, is_sim))
    
    def add_point_light(self, light: PLight):
        self.lights.append(light)
        
        light_arr = np.concatenate([light.pos.reshape((1, 3)), light.color.reshape((1, 3)).astype(np.float64) / 255.], axis=1)
        if self.lights_mat is None:
            self.lights_mat = light_arr
        else:
            self.lights_mat = np.concatenate([self.lights_mat, light_arr], axis=0)
    
    def _reverse_transform(self, rays: np.array, inst: ImplicitDrawable) -> np.array:
        rays = np.copy(rays)
        rays[:, :3] -= inst.Mtrans
        end = rays[:, :3] + rays[:, 3:]
        
        rays[:, :3] = (inst.Mrot.T @ rays[:, :3].T).T
        end = (inst.Mrot.T @ end.T).T
        
        rays[:, 3:] = end - rays[:, :3]
        
        end = rays[:, :3] + rays[:, 3:]
        rays[:, :3] /= inst.Mscale
        end /= inst.Mscale
        rays[:, 3:] = end - rays[:, :3]
        rays[:, 3:] /= np.linalg.norm(rays[:, 3:], axis=1)[:, np.newaxis]
        
        return rays
    
    def _process_GBuffer(self, rays: np.array, resol: int, refresh=False):
        if self.gbuffer is not None and not refresh:
            return
        
        _start = time.time()
        
        self.gbuffer = GBuffer(resol, resol)
        
        for model_name, model in self.models.items():
            for inst in self.drawables[model_name]:
                drays = self._reverse_transform(rays, inst)
                # drays = rays       
                
                ray_ts = utils.filter_rays_with_sphere(drays, r=1.2)
                drays[:, :3] = ray_ts[:, :3]
                
                depth = utils.recurv_inference_by_rays(drays, model, inst.latent, is_sim=inst.is_sim)
                # depth = utils.generate_inference_by_rays(drays, model, inst.latent, inst.is_sim)[:, -1]
                
                if depth is None:
                    depth = np.zeros(shape=(resol, resol))
                depth[depth >= 1.8] = np.inf
                
                depth = depth.reshape(-1, 1)
                depth = ray_ts[:, -1:] + depth
                
                depth_mat = depth.reshape((resol, resol))
                
                mask = depth_mat < self.gbuffer.depth_buffer
                
                _, normal_mat = utils.depth2normal_tangentspace(depth_mat)

                ## blend with g-buffer
                self.gbuffer.depth_buffer[mask]  = depth_mat[mask]
                self.gbuffer.normal_buffer[mask] = normal_mat[mask]
                self.gbuffer.raw_color[mask]     = inst.raw_color
        
        _end = time.time()
        
        print(f'[G-Buffer] generating takes {_end - _start:.3f}s')
    
    def deferred_shading(self, cam_pos: np.array, cam_dir: np.array, resol: int, modes: list=['depth'], refresh=False) -> list:
        '''
        modes = ['depth', 'normal', 'blinn-phong']
        '''
        
        raw_rays = utils.get_pinhole_rays(cam_pos, cam_dir, resol)

        self._process_GBuffer(raw_rays, resol, refresh)

        def depth_shade() -> np.array:
            depth_buffer = np.copy(self.gbuffer.depth_buffer)
            
            depth_buffer[depth_buffer == np.inf] = 0.
            
            style = 'gray_r'
            norm = Normalize()
            cmap = cm.get_cmap(style)
            
            frame = cmap(norm(depth_buffer))[:, :, :3]
            
            frame = (frame * 255.).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            
            return frame

        def normal_shade() -> np.array:
            normal_buffer = np.copy(self.gbuffer.normal_buffer)
            cam_pos_std = cam_pos / np.linalg.norm(cam_pos)
            
            normal_t = utils.get_rotation_matrix_from_points(cam_pos_std, np.array([0., 0., 1.]))
            normal_buffer = self.gbuffer.normal_buffer @ normal_t.T
            
            normal_buffer = (normal_buffer + 1) * 127.5
            normal_buffer = normal_buffer.clip(0, 255).astype(np.uint8)

            frame = cv2.cvtColor(normal_buffer, cv2.COLOR_RGB2BGR)
            
            frame[self.gbuffer.depth_buffer == np.inf] = np.array([255, 255, 255], dtype=np.uint8)
            
            return frame
        
        def phong_shade() -> np.array:
            ray_buffer = raw_rays.reshape((resol, resol, 6))
            
            cam_pos_std = cam_pos / np.linalg.norm(cam_pos)
            
            normal_t = utils.get_rotation_matrix_from_points(cam_pos_std, np.array([0., 0., 1.]))
            normal_buffer = self.gbuffer.normal_buffer @ normal_t.T
            
            frame = np.zeros_like(self.gbuffer.raw_color, dtype=np.uint8)
            
            shading_phong(ray_buffer, self.gbuffer.depth_buffer, normal_buffer, self.gbuffer.raw_color, self.lights_mat, frame)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # print(frame)
            return frame
            
        shade_mapper = {
            'depth': depth_shade,
            'normal': normal_shade,
            'blinn-phong': phong_shade
        }
        
        def log_profile(func, name):
            _start = time.time()
            rips = func()
            _end = time.time()
            print(f'[G-Buffer] - {name} - {_end - _start:.3f}s')
            return rips[::-1] # flip
        
        colors = {mode: log_profile(shade_mapper[mode], mode) for mode in modes}
        
        return colors

def render_snapshot(shot_path: str, impl_scene: ImplicitScene, resol: int, cam_pos: np.array, cam_dir: np.array):
    raw_rays = utils.get_pinhole_rays(cam_pos, cam_dir, resol)
    impl_scene._process_GBuffer(raw_rays, resol, refresh=True)

    depth_buffer = np.copy(impl_scene.gbuffer.depth_buffer)
    
    mask = depth_buffer == np.inf
    depth_buffer[mask] = np.min(depth_buffer, axis=None, keepdims=False)

    style = 'Greys'

    htmap = sns.heatmap(depth_buffer, cbar=True, xticklabels=False, yticklabels=False, mask=mask)
    htmap.get_figure().savefig(shot_path.replace('.png', f'_depth.png'), pad_inches=False, bbox_inches='tight', dpi=400)
    plt.close()
    
    depth_buffer[mask] = np.inf
    normal = utils.depth2normal_exp(depth_buffer, raw_rays.reshape((resol, resol, 6)))
    # normal = utils.depth2normal_worldspace(raw_rays, depth_buffer)
    
    frame = np.zeros_like(impl_scene.gbuffer.raw_color, dtype=np.uint8)
    shading_phong(
        raw_rays.reshape((resol, resol, 6)),
        impl_scene.gbuffer.depth_buffer,
        normal,
        impl_scene.gbuffer.raw_color,
        impl_scene.lights_mat,
        frame
    )
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(shot_path.replace('.png', f'_phong.png'), frame)
    
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    # Save the normal map to a file
    normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    normal_bgr[mask] = np.array([255., 255., 255.], dtype=np.uint8)
    cv2.imwrite(shot_path.replace('.png', f'_normal.png'), normal_bgr)
    
    normal = utils.depth2normal_worldspace_super(raw_rays, depth_buffer)
    frame = np.zeros_like(impl_scene.gbuffer.raw_color, dtype=np.uint8)
    shading_phong(
        raw_rays.reshape((resol, resol, 6)),
        impl_scene.gbuffer.depth_buffer,
        normal,
        impl_scene.gbuffer.raw_color,
        impl_scene.lights_mat,
        frame
    )
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(shot_path.replace('.png', f'_phong_2.png'), frame)
    
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    # Save the normal map to a file
    normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    normal_bgr[mask] = np.array([255., 255., 255.], dtype=np.uint8)
    cv2.imwrite(shot_path.replace('.png', f'_normal_2.png'), normal_bgr)
    
    
    if False:
        modes = ['depth', 'normal', 'blinn-phong']
        frames = impl_scene.deferred_shading(cam_pos, -cam_pos, resol, modes, True)
        
        for mode in modes:
            cv2.imwrite(shot_path.replace('.png', f'_{mode}.png'), frames[mode])

def render_tour_video(video_path: str, impl_scene: ImplicitScene, FPS: int, resol: int, frames: int, view_radius):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writers = {
        'depth': cv2.VideoWriter(video_path.replace('.mp4', '_depth.mp4'), fourcc, FPS, (resol, resol), True),
        'normal': cv2.VideoWriter(video_path.replace('.mp4', '_normal.mp4'), fourcc, FPS, (resol, resol), True),
        'blinn-phong': cv2.VideoWriter(video_path.replace('.mp4', '_phong.mp4'), fourcc, FPS, (resol, resol), True),
    }
    
    for cam_pos in tqdm(utils.get_equidistant_camera_angles(frames), desc='blending video', total=frames):
        cam_pos = view_radius * cam_pos
        
        frames = impl_scene.deferred_shading(cam_pos, -cam_pos, resol, list(video_writers.keys()), True)

        for shade_mode, video_writer in video_writers.items():
            video_writer.write(frames[shade_mode])
    
    for name in video_writers.keys():
        video_writers[name].release()
    
    del video_writers
    cv2.destroyAllWindows()

if __name__ == '__main__':
    impl_scene = ImplicitScene()
    
    model_names = ['T15_RDF', 'long_sleeve_upper_RDF', 'long_pants_RDF']
    
    models = []
    for mn in model_names:
        with open(os.path.join('configs', 'eval', mn+'.yml')) as stream:
            meta_params = yaml.safe_load(stream)
        
        model = RayDistanceField(**meta_params)
        model.load_state_dict(torch.load(meta_params['checkpoint_path']))
        model.cuda()
        model.eval()
        models.append(model)
    
    ## load scene with templates
    
    theta = np.radians(60)
    impl_scene.add_drawable(
        model_names[0], models[0], None,
        Mscale = [1., 1., 1.],
        Mrot = [
            [np.cos(theta), -np.sin(theta), 0.],
            [np.sin(theta), np.cos(theta), 0.],
            [0., 0., 1.],
        ],
        Mtrans = [0., 0.5, 0.]
    )
    
    impl_scene.add_drawable(
        model_names[1], models[1], None,
        Mscale = [2., 2., 2.],
        Mrot = [
            [np.cos(-theta), -np.sin(-theta), 0.],
            [np.sin(-theta), np.cos(-theta), .0],
            [0., 0., 1.]
        ],
        Mtrans = [0., 0., -3.2]
    )

    impl_scene.add_drawable(
        model_names[2], models[2], None,
        Mscale = [1.5, 1.5, 1.5],
        Mrot = [
            [1., 0., 0.],
            [0., np.cos(theta), -np.sin(theta)],
            [0., np.sin(theta), np.cos(theta)],
        ],
        Mtrans = [0., 0., 1.8]
    )

    
    impl_scene.add_point_light(PLight([3.,  0., 0.], [255, 0., 0.]))
    impl_scene.add_point_light(PLight([-3., 0., 0.], [0., 0., 255]))
    # impl_scene.add_point_light(nrender.PLight([0., -3., 0.], [166, 27, 41]))
    # impl_scene.add_point_light(nrender.PLight([0., 3., 0.], [49, 122, 167]))
    # impl_scene.add_point_light(nrender.PLight([3., 0., 0.], [252, 161, 6]))
    # impl_scene.add_point_light(nrender.PLight([-3., 0., 0.], [140, 194, 105]))
    
    ## gen video
    FPS    = 10
    resol  = 1024
    frames = 60
    radius = 8.
    video_path = 'tests/output/blend_render.mp4'
    
    render_tour_video(video_path, impl_scene, FPS, resol, frames, radius)
