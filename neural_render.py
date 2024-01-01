# -*- coding: utf-8 -*-

import os
import numpy as np
import utils
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import yaml
import torch
import cv2
from tqdm import tqdm

from networks.RayDFNet import RayDistanceField


class ImplicitDrawable:
    def __init__(
        self, embedding=None, 
        Mscale: np.array = [1., 1., 1.],
        Mrot: np.array = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        Mtrans: np.array = [0., 0., 0.]
    ):
        self.latent = embedding
        self.Mscale = np.array(Mscale)
        self.Mrot   = np.array(Mrot)
        self.Mtrans = np.array(Mtrans)
    
class ImplicitScene:
    def __init__(self):
        self.drawables: dict = {}  # {'T11_RDF': [ImplicitDrawable]}
        self.models: dict    = {}  # {'T11_RDF': NeuralNetworks}
    
    def add_drawable(
        self,
        model_name: str,
        model,
        embedding=None,
        Mscale: np.array = [1., 1., 1.],
        Mrot: np.array = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        Mtrans: np.array = [0., 0., 0.]
    ):
        if model_name not in self.models.keys():
            self.models[model_name] = model
            self.drawables[model_name] = []
        
        self.drawables[model_name].append(ImplicitDrawable(embedding, Mscale, Mrot, Mtrans))
    
    def _reverse_transform(self, rays: np.array, inst: ImplicitDrawable) -> np.array:
        rays = np.copy(rays)
        rays[:, :3] -= inst.Mtrans
        end = rays[:, :3] + rays[:, 3:]
        
        rays[:, :3] = (inst.Mrot.T @ rays[:, :3].T).T
        end = (inst.Mrot.T @ end.T).T
        
        rays[:, 3:] = end - rays[:, :3]
        
        return rays
    
    def draw_with_cam(self, cam_pos: np.array, cam_dir: np.array, resol: int) -> np.array:
        raw_rays = utils.get_pinhole_rays(cam_pos, cam_dir, resol)

        Frame     = np.zeros(shape=(resol * resol, 3))
        Z_buffer  = np.ones(shape=(resol * resol, 1)) * np.inf
        
        for model_name, model in self.models.items():
            for inst in self.drawables[model_name]:
                rays = self._reverse_transform(raw_rays, inst)           
                
                ray_ts = utils.filter_rays_with_sphere(rays, r=1.2)
                rays[:, :3] = ray_ts[:, :3]
                
                # depth = utils.recurv_inference_by_rays(rays, model, inst.latent)
                depth = utils.generate_inference_by_rays(rays, model, inst.latent)[:, -1]
                
                depth[depth >= 2.] = np.inf
                
                depth = depth.reshape(-1, 1)
                depth = ray_ts[:, -1:] + depth
                
                depth_mat = depth.reshape((resol, resol))
                depth_mat[depth_mat == np.inf] = 0.
                
                style = 'gray'

                norm = Normalize()
                cmap = cm.get_cmap(style)
                
                frame = cmap(norm(depth_mat))[:, :, :3]
                z_buffer = depth
                z_buffer[z_buffer == 0.] = np.inf

                blend_ind = (z_buffer < Z_buffer).flatten()
                frame = frame.reshape(-1, 3)
                
                Frame[blend_ind, :] = frame[blend_ind, :]
                Z_buffer[blend_ind, :] = z_buffer[blend_ind, :]

        return Frame.reshape((resol, resol, 3))
        
def render_tour_video(video_path: str, impl_scene: ImplicitScene, FPS: int, resol: int, frames: int, view_radius):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (resol, resol), True)
    
    for cam_pos in tqdm(utils.get_equidistant_camera_angles(frames), desc='blending video', total=frames):
        
        cam_pos = view_radius * cam_pos
        frame = impl_scene.draw_with_cam(cam_pos, -cam_pos, resol)
        
        frame = (frame * 255.).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        video_writer.write(frame)
        
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    impl_scene = ImplicitScene()
    
    model_names = ['T11_RDF', 'T15_RDF', 'airplane_RDF']
    
    models = []
    for mn in model_names:
        with open(os.path.join('configs', 'eval', mn+'.yml')) as stream:
            meta_params = yaml.safe_load(stream)
        
        model = RayDistanceField(**meta_params)
        model.load_state_dict(torch.load(meta_params['checkpoint_path']))
        model.cuda()
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
        Mscale = [1., 1., 1.],
        Mrot = [
            [1., 0., 0.],
            [0., np.cos(theta), -np.sin(theta)],
            [0., np.sin(theta), np.cos(theta)],
        ],
        Mtrans = [0., 0., 2.5]
    )

    impl_scene.add_drawable(
        model_names[2], models[2], None,
        Mscale = [1., 1., 1.],
        Mrot = [
            [1., 0., 0.],
            [0., np.cos(theta), -np.sin(theta)],
            [0., np.sin(theta), np.cos(theta)],
        ],
        Mtrans = [-1., 0., 0.]
    )
    
    ## gen video
    FPS    = 10
    resol  = 1024
    frames = 60
    radius = 7.
    video_path = 'tests/output/blend_render.mp4'
    
    render_tour_video(video_path, impl_scene, FPS, resol, frames, radius)
