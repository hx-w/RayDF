# -*- coding: utf-8 -*-

import math
import sys
import os

import trimesh
import numpy as np

sys.path.append('dataprocess')
from scan import (
    Scan,
    get_camera_transform,
    get_image_plane_positions
)

if sys.platform != 'win32':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

def get_equidistant_camera_angles(count):
    increment = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        theta = math.asin(-1 + 2 * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)
        yield phi, theta


def get_samples(mesh: trimesh.Trimesh, cam_pos: np.array, cam_dir: np.array, resol: int) -> np.array:
    cam_transf = get_camera_transform(cam_pos, cam_dir)
    scan = Scan(mesh, cam_transf, resol, False)
    image_pos = get_image_plane_positions(cam_pos, cam_dir, 0.1, 1, resol, resol)

    sample_rays = image_pos - camera_position
    sample_rays /= np.linalg.norm(sample_rays, axis=1)[:, np.newaxis]
    
    depth = -np.ones(shape=(resol*resol, 1)) # non hit = -1
    depth[scan.depth_buffer.flatten()!=0, :] = np.linalg.norm(scan.points - cam_pos, axis=1)[:, np.newaxis]
    
    # 转换球极坐标 theta, phi
    sphere_dirs = np.zeros(shape=(sample_rays.shape[0], 2))
    sphere_dirs[:, 0] = np.arctan2(sample_rays[:, 1], sample_rays[:, 0])
    sphere_dirs[:, 1] = np.arccos(sample_rays[:, 2])
    
    ray_origins = np.ones_like(sample_rays) * camera_position
    samples = np.concatenate([ray_origins, sphere_dirs, depth], axis=1)
    
    # (resol*resol, 6)
    return samples

if __name__ == '__main__':
    mesh = trimesh.load('test/mesh.obj')
    scan_resolution = 10
    camera_position = np.array([0.0, 10.0, 0.0])
    camera_diretion = np.array([0.0, -1.0, 0.0])
    
    get_samples(mesh, camera_position, camera_diretion, scan_resolution)
