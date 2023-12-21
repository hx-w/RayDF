# -*- coding: utf-8 -*-
import numpy as np

import trimesh
from scipy.io import loadmat
from preprocess import scale_to_unit_sphere


# test_mat_path = 'datasets/02747177/46e777a46aa76681f4fb4dee5181bee.mat'
test_mat_path = 'datasets/T11/n1_chenshun.mat'
# mesh_path = '../../DIF-Net/tooth_morphology/datasets/11_Outside/n1_chenshun/n1.obj'

if __name__ == '__main__':
    mat = loadmat(test_mat_path)['ray_depth']

    mat = mat[:100000, :]

    coords = mat[:, :3]
    # ball_dirs = mat[:, 3:5]
    dirs = mat[:, 3:6]
        
    depth = mat[:, -1].reshape(-1, 1)
    
    # dirs = np.concatenate([
    #     (np.sin(ball_dirs[:, 1]) * np.cos(ball_dirs[:, 0])).reshape(-1, 1),
    #     (np.sin(ball_dirs[:, 1]) * np.sin(ball_dirs[:, 0])).reshape(-1, 1),
    #     (np.cos(ball_dirs[:, 1])).reshape(-1, 1)
    # ], axis=1)
    

    depth[depth==2.0] = 0.0
    
    points = coords + dirs * depth
    
    trimesh.PointCloud(points).export('test/hide_sample_rec.ply')

    # ori_mesh = trimesh.load(mesh_path)
    # scale_to_unit_sphere(ori_mesh).export('test/hide_scaled_mesh.obj')


    import seaborn as sns
    
    resol = 256
    
    sub = depth[0*resol*resol: 1*resol*resol, :].reshape(resol, resol)
    
    sns.heatmap(sub, cmap='gray').get_figure().savefig('test/hide_sample.png')
