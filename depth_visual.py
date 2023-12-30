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
        up=[0, 1, 0] if (cam_dir[0] != 0 or cam_dir[2] != 0) else [0, 0, 1],
        width_px=resol,
        height_px=resol,
    )
    # normalize rays
    rays = rays.numpy().reshape((-1, 6))
    rays[:, 3:] /= np.linalg.norm(rays[:, 3:], axis=1)[:, np.newaxis]
    return rays

def generate_inference_by_rays(rays: np.array, model, embedding=None):
    inp_coords = torch.from_numpy(rays[:, :3]).reshape((1, rays.shape[0], 3)).cuda().float()
    inp_dirs = torch.from_numpy(rays[:, 3:]).reshape((1, rays.shape[0], 3)).cuda().float()
    
    if embedding is not None:
        depth = (
            model.inference(inp_coords, inp_dirs, embedding)
            .squeeze(1).detach().cpu().numpy().reshape((-1, 1))
        )
    else:
        depth = (
            model.get_template_field(inp_coords, inp_dirs)
            .squeeze(1).detach().cpu().numpy().reshape((-1, 1))
        )
    
    depth[depth[:, 0:] > 1.9] = 2.
    
    return np.concatenate([rays, depth], axis=1)
    

def generate_inference(cam_pos: np.array, cam_dir: np.array, model, resol: int, embedding=None):
    rays = get_pinhole_rays(cam_pos, cam_dir, resol) # (n, 6)
    
    return generate_inference_by_rays(rays, model, embedding)
    

def generate_scan(cam_pos: np.array, cam_dir: np.array, model, resol: int, filename: str=None, embedding=None):

    pixel_num = resol * resol
 
    rays = get_pinhole_rays(cam_pos, cam_dir, resol) # (n, 6)
    
    # inp_coords = torch.from_numpy(rays[:, :3]).reshape((1, pixel_num, 3))
    # inp_dirs = torch.from_numpy(rays[:, 3:]).reshape((1, pixel_num, 3))
    
    depth = recurv_inference_by_rays(rays, model, embedding)
    depth_mat = depth.reshape((resol, resol))
    
    style = 'gray'
    # plt.figure(figsize = (50, 50))
    htmap = sns.heatmap(depth_mat, cmap=style, cbar=False, xticklabels=False, yticklabels=False)
    
    if filename is not None:
        htmap.get_figure().savefig(filename, pad_inches=False, bbox_inches='tight')
        return

    canvas = htmap.get_figure().canvas
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(height, width, 3)
    
    plt.close()
    return image_array

def recurv_inference_by_rays(rays: np.array, model, embedding=None, thred: float=.3, stack_depth: int=0):
    depth = np.zeros(shape=(rays.shape[0], 1))
    
    samples = generate_inference_by_rays(rays, model, embedding)
    
    depth[samples[:, -1] >= 2.] = 2.
    depth[samples[:, -1] <= thred] = samples[samples[:, -1] <= thred][:, -1:]
    
    recurv_ind = np.logical_and(samples[:, -1] < 2., samples[:, -1] > thred)
    depth[recurv_ind] = thred
    
    if samples[recurv_ind].shape[0] > 0:
        sub_rays = rays[recurv_ind]
        sub_rays[:, :3] += thred * sub_rays[:, 3:]
        # print(f'recurv: {stack_depth} | {rays.shape[0]} => {sub_rays.shape[0]} with min {samples[recurv_ind][:, -1].min()}')
        depth[recurv_ind] += recurv_inference_by_rays(sub_rays, model, embedding, stack_depth=stack_depth+1)
    
    depth[depth[:, 0] > 2.] = 2.

    return depth
