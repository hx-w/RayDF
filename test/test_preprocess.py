# -*- coding: utf-8 -*-

import sys
import numpy as np
import trimesh

sys.path.append('dataprocess')

import dataprocess.preprocess as prep
import dataprocess.scan as scan

def points_to_spheres(points, radius=1):
    """
    Convert points into spheres for notebook-friendly display
    """
    return [
        trimesh.primitives.Sphere(radius=radius, center=pt) for pt in points
    ]

def test_sample_generator():
    mesh = trimesh.load('test/mesh.obj')
    
    sample_num = 5000000
    scan_resol = 400
    scan_count = sample_num // (scan_resol * scan_resol)
    print('scan_count:', scan_count)
    
    t_samples = []
    for theta, phi in prep.get_equidistant_camera_angles(scan_count):
        cam_transf = scan.get_camera_transform_looking_at_origin(phi, theta, 2 * 1)
        t_samples.append(prep.get_samples(mesh, cam_transf, scan_resol))

        assert(t_samples[-1].shape == (scan_resol*scan_resol, 6))

 
    t_samples = np.concatenate(t_samples, axis=0)

    print(t_samples.shape)


    ## render
    
    # ray_visualize = trimesh.load_path(
    #     np.hstack((ray_origins, ray_origins + sample_rays * depth)).reshape(-1, 2, 3)
    # )
    # ray_visualize.colors = np.ones(shape=(ray_origins.shape[0], 4)) * np.array([255, 0, 0, 255])
    # trimesh.Scene([ray_visualize, points_to_spheres(scan.points, radius=0.1)]).show()


if __name__ == '__main__':
    test_sample_generator()
