# -*- coding: utf-8 -*-


'''
brief:

'''


import gc
import sys
import os
from typing import List, Tuple
import logging
import configargparse

import trimesh
import numpy as np
from glob import glob
from scipy.io import savemat
from tqdm.contrib.concurrent import process_map
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm


logger = logging.getLogger('preprocess')
def setup_logger():
    global logger
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]

def sample_sphere_directions(amount):
    in_ball_cam_pos_1 = sample_uniform_points_in_unit_sphere(amount)
    in_ball_cam_pos_2 = sample_uniform_points_in_unit_sphere(amount)
    free_dirs = in_ball_cam_pos_2 - in_ball_cam_pos_1
    free_dirs /= np.linalg.norm(free_dirs, axis=1)[:, np.newaxis]

    return free_dirs
    

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
        max_scale = 1.2 * np.max([np.linalg.norm(mesh.vertices, axis=1).max() for mesh in meshes]) / 2.

    logger.info(f'max scale: {max_scale:.6f}')

    if np.abs(max_scale - 1.) > 1e-6:
        for ind in range(mesh_num):
            meshes[ind].apply_scale(1. / max_scale)

            try:
                meshes[ind].export(mesh_paths[ind])
            except Exception as e:
                logger.warn(f'SKIP file saving failed: {mesh_paths[ind]}')

    else:
        logger.info(f'geometry scaling passed')

    tags = [mesh_path.split(os.sep)[-2] for mesh_path in mesh_paths]
    return tags, meshes

'''
根据NeuralODF的采样策略
- 60% 单位球内的自由射线
- 40% 网格采样点球面射线 (以采样点为圆心，0.5为半径的球面)

由于所有几何都缩放至单位球内，根据经验，设采样的射线在以1.3为半径的球面
on_mesh_nums 为网格离散点个数

由于surf_rays的depth不一定为0.5 (因为可能有遮挡)，所以后续仍需要算surf_rays的depth
'''
def generate_sample_rays(mesh: trimesh.Trimesh, counts: int, radius: float=1.3, on_mesh_nums: int=8000) -> np.array:
    free_count = int(0.6 * counts)
    surf_count = counts - free_count
    
    ## freespace
    in_ball_cam_pos_1 = sample_uniform_points_in_unit_sphere(free_count)
    in_ball_cam_pos_2 = sample_uniform_points_in_unit_sphere(free_count)
    free_dirs = in_ball_cam_pos_2 - in_ball_cam_pos_1
    free_dirs /= np.linalg.norm(free_dirs, axis=1)[:, np.newaxis]
    free_oris = in_ball_cam_pos_1 * radius

    free_rays = np.concatenate([free_oris, free_dirs], axis=1)

    ## on-mesh
    on_surfs = mesh.sample(on_mesh_nums)
    each_free_count = surf_count // on_mesh_nums
    
    surf_rays = []
    for on_surf_point in on_surfs:
        each_free_dirs = sample_sphere_directions(each_free_count)
        
        surf_rays.append(np.concatenate([
            np.tile(on_surf_point, (each_free_count, 1)) - each_free_dirs * 1., each_free_dirs
        ], axis=1))
        
    surf_rays = np.concatenate(surf_rays, axis=0)
    
    return np.concatenate([free_rays, surf_rays], axis=0)

'''
无交点的射线长度设为2.0
'''   
def generate_sample_depth(scene: o3d.t.geometry.RaycastingScene, rays: np.array) -> np.array:
    riposta = scene.cast_rays(o3c.Tensor(rays.astype(np.float32)))
    
    depth = riposta['t_hit'].numpy().reshape(-1, 1)
    depth[depth == np.inf] = 2.0
    
    return depth

def sample_dataset(mesh: trimesh.Trimesh, sample_counts: int) -> np.array:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d))
    
    ## generate rays
    sample_rays  = generate_sample_rays(mesh, sample_counts, radius=1.3, on_mesh_nums=8000)
    
    ## generate depth
    sample_depth = generate_sample_depth(scene, sample_rays)

    samples = np.concatenate([sample_rays, sample_depth], axis=1)

    rand_inds = np.random.permutation(np.arange(samples.shape[0]))
    
    return samples[rand_inds, :]

if __name__ == '__main__':
    p = configargparse.ArgumentParser()
    p.add_argument('--split', '-s', type=str, required=True, help='`airplane`')
    p.add_argument('--data_dir', '-d', type=str, default='../../datasets')
    p.add_argument('--target', '-t', type=str, default='train')
    p.add_argument('--scale_factor', type=float, default=0.0, help='force scale, auto scale to unit ball if 0.0')
    p.add_argument('--sample_nums', '-n', type=int, default=1000000, help='rays sampled for each mesh')
    p.add_argument('--skip', action='store_true', default=False, help='skip sampled meshes')
    args = p.parse_args()
    
    setup_logger()

    mesh_paths = glob(os.path.join(args.data_dir, args.split) + '/*/*.obj')
    tags, meshes = load_and_unify(mesh_paths, args.scale_factor)

    os.makedirs(os.path.join('datasets', args.split), exist_ok=True)

    ## sample and save
    for tag, mesh in tqdm(zip(tags, meshes), 'sampling', total=len(tags)):
        ds_path = os.path.join('datasets', args.split, tag) + '.mat'
        
        if args.skip and os.path.isfile(ds_path):
            logger.info(f'SKIP {tag}')
            continue
        
        samples = sample_dataset(mesh, args.sample_nums)
    
        logger.info(f'SUCCESS {tag} with {samples.shape[0]} samples')
        
        savemat(ds_path, {'ray_depth': samples})
        
        gc.collect()

    ## save split
    with open(os.path.join('split', args.target, args.split)+'.txt', 'w') as ofh:
        ofh.write('\n'.join(tags))

    logger.info(f'> FINISH <')
