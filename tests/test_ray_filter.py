# -*- coding: utf-8 -*-

import os
import configargparse
from glob import glob
import math
import tqdm
import numpy as np
import open3d as o3d
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cv2

import utils as dv
import preprocess as prep

from tests import test_scan as ttscan


def get_equidistant_camera_angles(count):
    # increment = math.pi * (3 - math.sqrt(5))
    increment = (2. * np.pi) / count

    for i in range(count):
        phi = math.asin(-1 + 2 * i / (count - 1))
        # phi = 0.
        theta = ((i + 1) * increment) % (2 * math.pi)
        
        yield np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])   
        # yield theta, phi
    
if __name__ == '__main__':
    
    p = configargparse.ArgumentParser()
    p.add_argument('--split', '-s', type=str, required=True, help='`airplane`')
    p.add_argument('--data_dir', '-d', type=str, default='../../datasets')
    p.add_argument('--target', '-t', type=str, default='train')
    p.add_argument('--scale_factor', type=float, default=0.0, help='force scale, auto scale to unit ball if 0.0')
    p.add_argument('--sample_nums', '-n', type=int, default=1000000, help='rays sampled for each mesh')
    args = p.parse_args()
    
    prep.setup_logger()
    
    mesh_paths = glob(os.path.join(args.data_dir, args.split) + '/*/*.obj')
    ## for test, only 1 mesh
    mesh_paths = mesh_paths[-2:-1]
    
    tags, meshes = ttscan.load_and_unify(mesh_paths, args.scale_factor)

    radius = 5.
    resol  = 1024
    FPS    = 24
    FRAMES = 20
    
    # sample and save
    for tag, mesh in zip(tags, meshes):
        png_path = os.path.join('tests/output', f'{tag}_filter')
        
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d))
        
        def _save_heatmap(mat: np.array):
            mat[mat >= np.inf] = 0.

            style = 'gray'
            cmap = sns.cubehelix_palette(start=0.0, gamma=0.8, as_cmap=True)
            
            norm = Normalize()
            cmap = cm.get_cmap(style)
            image_arr = cmap(norm(mat))[:, :, :3]
            
            return image_arr

        def _save_video(frames: int, video_path: str):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (resol, resol), True)

            for cam_pos in tqdm(get_equidistant_camera_angles(frames), desc='gen video', total=frames):
                cam_pos = radius * cam_pos
                rays = dv.get_pinhole_rays(cam_pos, -cam_pos, resol)
                
                ray_ts = dv.filter_rays_with_sphere(rays, r=1.3)
                rays[:, :3] = ray_ts[:, :3]
                
                depth_rec = ttscan.generate_sample_depth_recursive(scene, rays, 0.3)
                depth_rec[depth_rec >= 2.] = np.inf

                depth_rec = ray_ts[:, -1:] + depth_rec

                img = _save_heatmap(depth_rec.reshape((resol, resol)))
                _save_heatmap(depth_rec.reshape((resol, resol)))
                
                # print(img.shape)
                img = (img * 255.).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # print(img)
                
                video_writer.write(img)
                
            video_writer.release()
            cv2.destroyAllWindows()

            # os.remove('.temp.png')
        _save_video(FRAMES, f'{png_path}.mp4')
            
    pass