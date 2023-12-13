# -*- coding: utf-8 -*-
'''
给定一个网格，采样n个射线，以及射线的深度，如果射线与网格不相交，深度为1

'ray_depth': shape=(n, 6)

| cam_x | cam_y | cam_z | ray_theta | ray_phi | depth |
'''


import math
import sys
import os
from typing import List

import trimesh
import numpy as np
import configargparse
from glob import glob
from scipy.io import savemat
from tqdm.contrib.concurrent import process_map
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm


if sys.platform != 'win32':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

def get_equidistant_camera_angles(count):
    increment = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        theta = math.asin(-1 + 2 * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)
        yield theta, phi

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

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

def sample_on_sphere_directions(amount):
    unit_sphere_dirs = np.random.uniform(-1, 1, size=(amount * 2, 3))
    unit_sphere_dirs = unit_sphere_dirs[np.linalg.norm(unit_sphere_dirs, axis=1) < 0.1]

    dirs_available = unit_sphere_dirs.shape[0]
    if dirs_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:dirs_available, :] = unit_sphere_dirs
        result[dirs_available:, :] = sample_uniform_points_in_unit_sphere(amount - dirs_available)
        return result
    else:
        return unit_sphere_dirs[:amount, :]

'''
每个相机参数的采样结果
'''
def get_samples(mesh: trimesh.Trimesh, cam_pos: np.array, cam_dir: np.array, resol: int) -> np.array:
    scene = o3d.t.geometry.RaycastingScene()
    
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d))
    
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=cam_pos + cam_dir,
        eye=cam_pos,
        up=[0, 1, 0],
        width_px=resol,
        height_px=resol,
    )
    # normalize rays
    rays = rays.numpy().reshape((-1, 6))
    rays[:, 3:] /= np.linalg.norm(rays[:, 3:], axis=1)[:, np.newaxis]
    rays = o3c.Tensor(rays.reshape((resol, resol, 6)))
    
    ans = scene.cast_rays(rays)
    
    depth = ans['t_hit'].numpy().reshape((-1, 1))
    depth[depth == np.inf] = 2.0
    
    rays = rays.numpy().reshape((-1, 6))
    
    # 转换球极坐标 theta, phi
    sphere_dirs = np.zeros(shape=(rays.shape[0], 2))
    sphere_dirs[:, 0] = np.arctan2(rays[:, 4], rays[:, 3])
    sphere_dirs[:, 1] = np.arccos(rays[:, 5])
    
    samples = np.concatenate([rays[:, :3], sphere_dirs, depth], axis=1)
    
    # (resol*resol, 6)
    return samples

def fetch_files(data_dir: str, split_tag: str) -> List[str]:
    return glob(os.path.join(data_dir, split_tag) + '/*/*.obj')

def sample_data(file_path: str):
    print('process:', file_path)
    
    # mesh scaled
    mesh = scale_to_unit_sphere(trimesh.load(file_path))
    split_tag = args.split

    tag = os.path.split(os.path.split(file_path)[0])[-1]
    os.makedirs(os.path.join('datasets', split_tag), exist_ok=True)

    mat_path = os.path.join('datasets', split_tag, tag)+'.mat'

    if True or not os.path.isfile(mat_path):
        scan_resol = 256 # same as ODF
        scan_count = 150
        t_samples = []

        # on-sphere samplings
        # in_ball_cam_pos_1 = sample_uniform_points_in_unit_sphere(scan_count // 2)
        # in_ball_cam_pos_2 = sample_uniform_points_in_unit_sphere(scan_count // 2)
        # in_ball_dir = in_ball_cam_pos_2 - in_ball_cam_pos_1
        # in_ball_dir /= np.linalg.norm(in_ball_dir, axis=1)[:, np.newaxis]
        
        # for ind in range(in_ball_dir.shape[0]):
        #     cam_pos = in_ball_dir[ind, :]
        #     t_samples.append(get_samples(mesh, cam_pos, -cam_pos, scan_resol))
        
        for theta, phi in get_equidistant_camera_angles(scan_count // 2):
            # 圆球上的方向向量
            cam_pos = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]) * 1.3
            t_samples.append(get_samples(mesh, cam_pos, -cam_pos, scan_resol))
        
        # in-ball samplings
        in_ball_cam_pos_1 = sample_uniform_points_in_unit_sphere(scan_count // 2)
        in_ball_cam_pos_2 = sample_uniform_points_in_unit_sphere(scan_count // 2)
        in_ball_dir = in_ball_cam_pos_2 - in_ball_cam_pos_1
        in_ball_dir /= np.linalg.norm(in_ball_dir, axis=1)[:, np.newaxis]
        
        for ind in range(in_ball_dir.shape[0]):
            cam_pos = in_ball_cam_pos_1[ind, :]
            cam_dir = in_ball_dir[ind, :]
            samples = get_samples(mesh, cam_pos, cam_dir, scan_resol)
            # random sample resol x resol -> resol
            rand_inds = np.random.choice(samples.shape[0], scan_resol * scan_resol // 5)
            t_samples.append(samples[rand_inds, :])

        t_samples = np.concatenate(t_samples, axis=0)

        rand_inds = np.random.permutation(np.arange(t_samples.shape[0]))
        
        print('finish:', file_path.split('/')[-1], 'with samples', t_samples.shape[0])

        savemat(mat_path, { 'ray_depth': t_samples[rand_inds, :] })
        # savemat(mat_path, {'ray_depth': t_samples})
        
if __name__ == '__main__':
    p = configargparse.ArgumentParser()
    p.add_argument('--split', '-s', type=str, required=True, help='`11_Outside`')
    p.add_argument('--data', '-d', type=str, default='../../DIF-Net/tooth_morphology/datasets')
    p.add_argument('--target', '-t', type=str, default='train')
    args = p.parse_args()
    
    files = fetch_files(args.data, f'{args.split[1:]}_Outside')

    # process_map(sample_data, files, max_workers=5, chunksize=1)
    for file in tqdm(files):
        sample_data(file)

    file_tags = [
        os.path.split(os.path.split(fp)[0])[-1] for fp in files
    ]
    with open(os.path.join('split', args.target, args.split)+'.txt', 'w') as ofh:
        ofh.write('\n'.join(file_tags))
