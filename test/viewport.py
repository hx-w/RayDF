# -*- coding: utf-8 -*-

import os
from glob import glob
from tqdm import tqdm
import trimesh
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import preprocess as prep


base_dir = '../../DIF-Net/tooth_morphology/datasets/15_Outside'

all_files = glob(base_dir + '/*/*.obj')

resol = 256

for filepath in tqdm(all_files):
    tag = filepath.split('/')[-2]
    mesh = trimesh.load(filepath)
    
    mesh = prep.scale_to_unit_sphere(mesh)

    cam_coeffs = [
        np.array([1.3, 0.0, 0.0]),
        np.array([0.0, 1.3, 0.0]),
        np.array([0.0, 0.0, 1.3]),
    ]
    
    for ind, cam_pos in enumerate(cam_coeffs):
        samples = prep.get_samples(mesh, cam_pos, -cam_pos, resol)

        depth = samples[:, -1].reshape((resol, resol))

        style = 'gray'
        htmap = sns.heatmap(depth, cmap=style, cbar=True, xticklabels=False, yticklabels=False)
        
        filename = os.path.join('test/viewports', f'{tag}_{ind}.png')
        htmap.get_figure().savefig(filename, pad_inches=False, bbox_inches='tight')
    
        plt.close()
