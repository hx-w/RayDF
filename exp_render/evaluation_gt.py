# -*- coding: utf-8 -*-

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from typing import List, Tuple
import trimesh
import yaml
import numpy as np
import cv2

import torch
import configargparse

from glob import glob
import open3d as o3d
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pyrender

import preprocess as prep
import utils as dv
from render import neural_render, shade

logger = prep.logger


'''
取所有几何中最长的轴向边长的倒数为缩放因数，同比例缩放所有几何
保证所有几何的对齐关系不变
'''
def load_and_unify(mesh_paths: List[str], scale_factor: float=0.0) -> Tuple[trimesh.Trimesh]:
    mesh_num = len(mesh_paths)
    meshes =  [trimesh.load(mpath, force='mesh') for mpath in tqdm(mesh_paths, desc='loading')]
    
    for mesh in meshes:
        mesh.vertices -= mesh.centroid
    # to unit ball
    if scale_factor:
        max_scale = scale_factor
    else:
        # max / 2
        max_scale = 1. * np.max([np.linalg.norm(mesh.vertices, axis=1).max() for mesh in meshes])

    logger.info(f'max scale: {max_scale:.6f}')

    for ind in range(mesh_num):
        meshes[ind].apply_scale(1. / max_scale)

    tags = [mesh_path.split(os.sep)[-1].split('.')[0] for mesh_path in mesh_paths]
    return tags, meshes


if __name__ == '__main__':
    p = configargparse.ArgumentParser()
    p.add_argument('--mesh', '-m', type=str, required=True, help='`airplane`')
    p.add_argument('--scale_factor', type=float, default=0.0, help='force scale, auto scale to unit ball if 0.0')
    args = p.parse_args()
    
    prep.setup_logger()
    
    tags, meshes = load_and_unify([args.mesh], args.scale_factor)

    radius = 1.3
    resol  = 1024
    
    locs = [
        np.array([radius, 0.5, 0.2]),
        np.array([0., -0.5, radius])
    ]

    ## sample and save
    for tag, mesh in tqdm(zip(tags, meshes), 'sampling', total=len(tags)):
        png_path = os.path.join('exp_render/eval/', f'{tag}_gt')
        os.makedirs(png_path, exist_ok=True)
        
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d))
        
        for idx, loc in enumerate(locs):
            cam_pos = np.array(loc)
            rays = dv.get_pinhole_rays(cam_pos, -cam_pos, resol)
            
            depth_raw, normal_raw = prep.generate_sample_depth(scene, rays, return_normal=True)
            inv_mask = (depth_raw == 2.).reshape((resol, resol))
            depth_raw[depth_raw == 2.] = np.min(depth_raw, axis=None, keepdims=False)
            
            depth_mat = np.copy(depth_raw).reshape((resol, resol))
            depth_mat[inv_mask] = np.inf
            
            def _save_heatmap(mat: np.array, path: str):
                style = 'Greys'
                # cmap = sns.cubehelix_palette(start=0.0, gamma=0.8, as_cmap=True)

                htmap = sns.heatmap(mat, cbar=True, xticklabels=False, yticklabels=False, mask=inv_mask)
                htmap.get_figure().savefig(path, pad_inches=False, bbox_inches='tight', dpi=400)
                
                # canvas = htmap.get_figure().canvas
                # width, height = canvas.get_width_height()
                # image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                # norm = Normalize()
                # cmap = cm.get_cmap(style)
                # image_arr = cmap(norm(mat))[:, :, :3]

                test_normal = dv.depth2normal_exp(depth_mat, rays.reshape((resol, resol, 6)))
                normal = (test_normal.reshape((resol, resol, 3)) + 1) * 127.5
                normal = normal.clip(0, 255).astype(np.uint8)

                # Save the normal map to a file
                normal_image = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
                normal_image[inv_mask] = np.array([255., 255., 255.], dtype=np.uint8)
                cv2.imwrite(path.replace('depth', 'test_normal'), normal_image)
                
                # normal_image, _ = dv.depth2normal_tangentspace(mat)
                # normal_image[inv_mask] = np.array([255., 255., 255.], dtype=np.uint8)
                normal = (normal_raw.reshape((resol, resol, 3)) + 1) * 127.5
                normal = normal.clip(0, 255).astype(np.uint8)

                # Save the normal map to a file
                normal_image = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
                normal_image[inv_mask] = np.array([255., 255., 255.], dtype=np.uint8)
                
                impl_scene = neural_render.ImplicitScene()
                impl_scene.add_point_light(neural_render.PLight([3.,  0.5, -2.], [225, 225, 225]))
                impl_scene.add_point_light(neural_render.PLight([-3., 0.5, 2.], [225., 225., 225]))
                
                depth_raw[inv_mask.flatten()] = np.inf
                raw_color = np.ones(shape=(resol, resol, 3), dtype=np.uint8) * np.array([100, 100, 100], dtype=np.uint8) # 255
                frame = np.zeros_like(raw_color, dtype=np.uint8)
                shade.shading_phong(
                    rays.reshape((resol, resol, 6)),
                    depth_raw.reshape((resol, resol)),
                    normal_raw.reshape((resol, resol, 3)),
                    raw_color,
                    impl_scene.lights_mat,
                    frame
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                
                plt.close()
                
                cv2.imwrite(path.replace('depth', 'normal'), normal_image)
                cv2.imwrite(path.replace('depth', 'phong'), frame)
            
            _save_heatmap(depth_raw.reshape((resol, resol)), os.path.join(png_path, f'{tag}_{idx}_depth.png'))

            # print(points.max(), points.min())
        