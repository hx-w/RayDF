# -*- coding: utf-8 -*-

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from dataprocess.scan import get_image_plane_positions


def generate_scan(cam_pos: np.array, cam_dir: np.array, model, resol: int):
    image_pos = get_image_plane_positions(cam_pos, cam_dir, 0.1, 1, resol, resol)

    pixel_num = resol * resol
    sample_rays = image_pos - cam_pos
    sample_rays /= np.linalg.norm(sample_rays, axis=1)[:, np.newaxis]
    
    # to theta, phi
    inp_dirs = np.zeros(shape=(pixel_num, 2))
    inp_dirs[:, 0] = np.arctan2(sample_rays[:, 1], sample_rays[:, 0])
    inp_dirs[:, 1] = np.arccos(sample_rays[:, 2])
    
    inp_coords = torch.from_numpy(np.ones_like(sample_rays) * cam_pos).reshape((1, pixel_num, 3)).cuda().float()
    inp_dirs = torch.from_numpy(inp_dirs).reshape((1, pixel_num, 2)).cuda().float()
    
    depth_mat = (
        model.get_template_field(inp_coords, inp_dirs)
        .squeeze(1).detach().cpu().numpy().reshape((resol, resol))
    )

    style = 'coolwarm'
    plt.figure(figsize = (50, 50))
    htmap = sns.heatmap(depth_mat, cmap=style, cbar=False, xticklabels=False, yticklabels=False)
    # htmap.get_figure().savefig(filename, pad_inches=False, bbox_inches='tight')
    canvas = htmap.get_figure().canvas
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(height, width, 3)
    
    plt.close()
    return image_array