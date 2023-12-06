# -*- coding: utf-8 -*-

import sys
import numpy as np
import trimesh

sys.path.append('dataprocess')

import dataprocess.preprocess as prep

def points_to_spheres(points, radius=1):
    """
    Convert points into spheres for notebook-friendly display
    """
    return [
        trimesh.primitives.Sphere(radius=radius, center=pt) for pt in points
    ]

def test_sample_generator():
    mesh = trimesh.load('test/mesh.obj')
    scan_resolution = 10
    camera_position = np.array([0.0, 10.0, 0.0])
    camera_diretion = np.array([0.0, -1.0, 0.0])
    
    samples = prep.get_samples(mesh, camera_position, camera_diretion, scan_resolution)

    assert(samples.shape == (scan_resolution*scan_resolution, 6))

    ## render
    
    # ray_visualize = trimesh.load_path(
    #     np.hstack((ray_origins, ray_origins + sample_rays * depth)).reshape(-1, 2, 3)
    # )
    # ray_visualize.colors = np.ones(shape=(ray_origins.shape[0], 4)) * np.array([255, 0, 0, 255])
    # trimesh.Scene([ray_visualize, points_to_spheres(scan.points, radius=0.1)]).show()