# -*- coding: utf-8 -*-

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import yaml
import numpy as np

import trimesh
import torch
import configargparse
# from calculate_chamfer_distance import compute_recon_error
from networks.SimRayDFNet import SimRayDistanceField
from networks.SDFNet import SDFNet

import render.neural_render as nrender
import utils
from metrics import chamfer, emd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

p = configargparse.ArgumentParser()
p.add_argument('--config', required=True, help='Evaluation configuration')

# load configs
opt = p.parse_args()
with open(os.path.join(opt.config),'r') as stream:
    meta_params = yaml.safe_load(stream)

# define Net
if meta_params['net'] == 'sim_RDF':
    model = SimRayDistanceField(**meta_params)
else:
    model = SDFNet(**meta_params)

model.load_state_dict(torch.load(meta_params['checkpoint_path']))

# The network should be fixed for evaluation.
for param in model.res1.parameters():
    param.requires_grad = False
for param in model.res2.parameters():
    param.requires_grad = False

model.cuda()

# create save path
root_path = os.path.join('exp_render/eval', meta_params['experiment_name'])
os.makedirs(root_path, exist_ok=True)

impl_scene = nrender.ImplicitScene()

model_name = meta_params['experiment_name']
displace = 1
impl_scene.add_drawable(
    model_name,
    model,
    # Mtrans=[1.4 * (displace // 2) * ((-1) ** (displace % 2)), 0., 0.],
    is_sim=True
)

impl_scene.add_point_light(nrender.PLight([3.,  0.5, -2.], [225, 225, 225]))
impl_scene.add_point_light(nrender.PLight([-3., 0.5, 2.], [225., 225., 225]))

# nrender.render_tour_video(
#     os.path.join(root_path, f'blend_{model_name}.mp4'),
#     impl_scene,
#     FPS=10,
#     resol=512,
#     frames=90,
#     # view_radius=(displace // 2)+1.2
#     view_radius=1.6
# )

radius = 1.3
resol = 512
locs = [
    np.array([radius, 0.5, 0.2]),
    np.array([0., -0.5, radius])
]

for idx, loc in enumerate(locs):
    nrender.render_snapshot(
        os.path.join(root_path, f'{model_name}_{idx}.png'),
        impl_scene,
        resol,
        cam_pos=np.array(loc),
        cam_dir=-np.array(loc)
    )
    
## EVAL - pointcloud

points = utils.generate_pointcloud_super(model, 50000, 1.2, is_sim=True)

gt_mesh = trimesh.load(meta_params['source_mesh'], force='mesh')
## align
gt_mesh.vertices -= gt_mesh.centroid
max_scale = 1. * np.max([np.linalg.norm(mesh.vertices, axis=1).max() for mesh in [gt_mesh]]) 
gt_mesh.apply_scale(1. / max_scale)

cf = chamfer.compute_trimesh_chamfer(gt_mesh, points, 0, 1, num_mesh_samples=30000)
emd = emd.compute_trimesh_emd(gt_mesh, points, 0, 1, num_mesh_samples=500)

print(f'[{meta_params["experiment_name"]}] chamfer: {cf}   emd: {emd}')

