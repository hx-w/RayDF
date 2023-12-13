# -*- coding: utf-8 -*-

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.core as o3c


def get_pinhole_rays(cam_pos: np.array, cam_dir: np.array, resol: int):
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
    return rays

def generate_scan(cam_pos: np.array, cam_dir: np.array, model, resol: int, filename: str=None, embedding=None):

    pixel_num = resol * resol
 
    rays = get_pinhole_rays(cam_pos, cam_dir, resol) # (n, 6)
    
    # to theta, phi
    inp_dirs = np.zeros(shape=(pixel_num, 2))
    inp_dirs[:, 0] = np.arctan2(rays[:, 4], rays[:, 3])
    inp_dirs[:, 1] = np.arccos(rays[:, 5])
    
    inp_coords = torch.from_numpy(rays[:, :3]).reshape((1, pixel_num, 3)).cuda().float()
    inp_dirs = torch.from_numpy(inp_dirs).reshape((1, pixel_num, 2)).cuda().float()
    
    if embedding is not None:
        depth_mat = (
            model.inference(inp_coords, inp_dirs, embedding)
            .squeeze(1).detach().cpu().numpy().reshape((resol, resol))
        )
    else:
        depth_mat = (
            model.get_template_field(inp_coords, inp_dirs)
            .squeeze(1).detach().cpu().numpy().reshape((resol, resol))
        )

    style = 'coolwarm'
    # plt.figure(figsize = (50, 50))
    htmap = sns.heatmap(depth_mat, cmap=style, cbar=False, xticklabels=False, yticklabels=False)
    
    if filename is not None:
        htmap.get_figure().savefig(filename, pad_inches=False, bbox_inches='tight')

    canvas = htmap.get_figure().canvas
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(height, width, 3)
    
    plt.close()
    return image_array