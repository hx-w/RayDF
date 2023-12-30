# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trimesh
import yaml
import configargparse
import numpy as np

import torch
from networks.RayDFNet import RayDistanceField
from depth_visual import generate_scan, recurv_inference_by_rays
import preprocess as prep

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

radius = 1.3

p = configargparse.ArgumentParser()

p.add_argument('--config', required=True, help='generation configuration')

# load configs
opt = p.parse_args()
with open(os.path.join(opt.config),'r') as stream:
    meta_params = yaml.safe_load(stream)

# define DIF-Net
model = RayDistanceField(**meta_params)
model.load_state_dict(torch.load(meta_params['checkpoint_path']))
model.cuda()

# create save path
root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name'])

# generate meshes with color-coded coordinates

template_path = os.path.join(root_path, 'template')
os.makedirs(template_path, exist_ok=True)

generate_scan(
    cam_pos=np.array([radius, 0.0, 0.0]),
    cam_dir=np.array([-radius, 0.0, 0.0]),
    model=model,
    resol=256,
    filename=os.path.join(template_path, 'view_X.png'),
)
generate_scan(
    cam_pos=np.array([0.0, radius, 0.0]),
    cam_dir=np.array([0.0, -radius, 0.0]),
    model=model,
    resol=256,
    filename=os.path.join(template_path, 'view_Y.png'),
)
generate_scan(
    cam_pos=np.array([0.0, 0.0, radius]),
    cam_dir=np.array([0.0, 0.0, -radius]),
    model=model,
    resol=256,
    filename=os.path.join(template_path, 'view_Z.png'),
)



# reconstruct pointcloud
counts = 50000            
in_ball_cam_pos_1 = prep.sample_uniform_points_in_unit_sphere(counts)
in_ball_cam_pos_2 = prep.sample_uniform_points_in_unit_sphere(counts)
free_dir = in_ball_cam_pos_2 - in_ball_cam_pos_1
free_dir /= np.linalg.norm(free_dir, axis=1)[:, np.newaxis]
free_ori = in_ball_cam_pos_1 * 1.3

rays = np.concatenate([free_ori, free_dir], axis=1)
depth = recurv_inference_by_rays(rays, model)
# t_samples = generate_inference_by_rays(rays, model)
t_samples = np.concatenate([rays, depth], axis=1)

t_samples = t_samples[t_samples[:, -1] < 2.0]
coords, dirs, depth = t_samples[:, :3], t_samples[:, 3:-1], t_samples[:, -1].reshape(-1, 1)

points = coords + dirs * depth

trimesh.PointCloud(points).export(os.path.join(template_path, 'template.ply'))
