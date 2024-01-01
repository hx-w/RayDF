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
from utils import generate_pointcloud_super, generate_tour_video_super
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

tag = meta_params['experiment_name']
template_path = os.path.join(root_path, 'template')
os.makedirs(template_path, exist_ok=True)

# generate_scan(
#     cam_pos=np.array([radius, 0.0, 0.0]),
#     cam_dir=np.array([-radius, 0.0, 0.0]),
#     model=model,
#     resol=256,
#     filename=os.path.join(template_path, 'view_X.png'),
# )
# generate_scan(
#     cam_pos=np.array([0.0, radius, 0.0]),
#     cam_dir=np.array([0.0, -radius, 0.0]),
#     model=model,
#     resol=256,
#     filename=os.path.join(template_path, 'view_Y.png'),
# )
# generate_scan(
#     cam_pos=np.array([0.0, 0.0, radius]),
#     cam_dir=np.array([0.0, 0.0, -radius]),
#     model=model,
#     resol=256,
#     filename=os.path.join(template_path, 'view_Z.png'),
# )

generate_tour_video_super(model, radius=2., FPS=24, frames=90, resol=512, filename=os.path.join(template_path, f'{tag}.mp4'))

points = generate_pointcloud_super(model, 50000, 1.3, None, True)

trimesh.PointCloud(points).export(os.path.join(template_path, f'{tag}.ply'))
