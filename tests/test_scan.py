# -*- coding: utf-8 -*-
'''
`python3 test/test_scan.py`
'''

import os
from typing import List, Tuple
import math

import trimesh
import configargparse
import numpy as np
from glob import glob
import open3d as o3d
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import preprocess as prep
import utils as dv

logger = prep.logger

def generate_sample_depth_recursive(scene: o3d.t.geometry.RaycastingScene, rays: np.array, thred: float=0.3, stack_depth: int=0) -> np.array:

    depth = prep.generate_sample_depth(scene, rays)
    samples = np.concatenate([rays, depth], axis=1)
    
    depth[samples[:, -1] >= 2.] = 2.
    depth[samples[:, -1] <= thred] = samples[samples[:, -1] <= thred][:, -1:]
    
    recurv_ind = np.logical_and(samples[:, -1] < 2., samples[:, -1] > thred)
    depth[recurv_ind] = thred
    
    logger.debug(f'stack {stack_depth} with rays {rays.shape[0]} => {samples[recurv_ind].shape[0]}')
    if samples[recurv_ind].shape[0] > 0:
        sub_rays = rays[recurv_ind]
        sub_rays[:, :3] += thred * sub_rays[:, 3:]
        # print(f'recurv: {stack_depth} | {rays.shape[0]} => {sub_rays.shape[0]} with min {samples[recurv_ind][:, -1].min()}')
        depth[recurv_ind] += generate_sample_depth_recursive(scene, sub_rays, thred, stack_depth+1)
    
    depth[depth[:, 0] >= 2.] = np.inf

    return depth

'''
取所有几何中最长的轴向边长的倒数为缩放因数，同比例缩放所有几何
保证所有几何的对齐关系不变
'''
def load_and_unify(mesh_paths: List[str], scale_factor: float=0.0) -> Tuple[trimesh.Trimesh]:
    mesh_num = len(mesh_paths)
    meshes =  [trimesh.load(mpath, force='mesh') for mpath in tqdm(mesh_paths, desc='loading')]
    
    # to unit ball
    if scale_factor:
        max_scale = scale_factor
    else:
        # max / 2
        max_scale = 1. * np.max([np.linalg.norm(mesh.vertices, axis=1).max() for mesh in meshes])

    logger.info(f'max scale: {max_scale:.6f}')

    for ind in range(mesh_num):
        meshes[ind].apply_scale(1. / max_scale)

    tags = [mesh_path.split(os.sep)[-2] for mesh_path in mesh_paths]
    return tags, meshes


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
    
    tags, meshes = load_and_unify(mesh_paths, args.scale_factor)

    radius = 1.3
    resol  = 1024

    ## sample and save
    for tag, mesh in tqdm(zip(tags, meshes), 'sampling', total=len(tags)):
        png_path = os.path.join('tests/output', f'{tag}')
        
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d))
        
        cam_pos = np.array([0, radius, 0.])
        rays = dv.get_pinhole_rays(cam_pos, -cam_pos, resol)
        
        depth_rec = generate_sample_depth_recursive(scene, rays, 0.3)
        depth_raw = prep.generate_sample_depth(scene, rays)
        
        def _save_heatmap(mat: np.array, path: str):
            style = 'gray_r'
            cmap = sns.cubehelix_palette(start=0.0, gamma=0.8, as_cmap=True)

            htmap = sns.heatmap(mat, cmap=style, cbar=True, xticklabels=False, yticklabels=False)
            htmap.get_figure().savefig(path, pad_inches=False, bbox_inches='tight')
            
            # canvas = htmap.get_figure().canvas
            # width, height = canvas.get_width_height()
            # image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            norm = Normalize()
            cmap = cm.get_cmap(style)
            image_arr = cmap(norm(mat))[:, :, :3]
            
            plt.close()
            
        _save_heatmap(depth_rec.reshape((resol, resol)), png_path+'_recursive.png')
        _save_heatmap(depth_raw.reshape((resol, resol)), png_path+'_direct.png')
