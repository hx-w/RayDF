# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Evaluation script for DIF-Net.
'''

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import yaml
import io
import numpy as np
import dataset, training_loop
from networks import loss, modules, meta_modules

import torch
from torch.utils.data import DataLoader
import configargparse
from torch import nn
from networks.RayDFNet import RayDistanceField
from networks.ODFNet import OmniDistanceField
# from calculate_chamfer_distance import compute_recon_error

import render.neural_render as nrender


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

p = configargparse.ArgumentParser()
p.add_argument('--config', required=True, help='Evaluation configuration')

# load configs
opt = p.parse_args()
with open(os.path.join(opt.config),'r') as stream:
    meta_params = yaml.safe_load(stream)


# define Net
if meta_params['net'] == 'RayDistanceField':
    model = RayDistanceField(**meta_params)
else:
    model = OmniDistanceField(**meta_params)

model.load_state_dict(torch.load(meta_params['checkpoint_path']))

# The network should be fixed for evaluation.
if meta_params['net'] == 'RayDistanceField':
    for param in model.template_field.parameters():
        param.requires_grad = False
else:
    for param in model.forward_net.parameters():
        param.requires_grad = False

for param in model.hyper_net.parameters():
    param.requires_grad = False

model.cuda()

# create save path
root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name'])
os.makedirs(root_path, exist_ok=True)

# load names for evaluation subjects
with open(meta_params['eval_split'],'r') as file:
    all_names = file.read().split('\n')

impl_scene = nrender.ImplicitScene()
model_name = meta_params['experiment_name']

displace = 1

# optimize latent code for each test subject
for file in all_names:
    save_path = os.path.join(root_path, file)

    # load ground truth data
    sdf_dataset = dataset.RayDepthMulti(root_dir=[os.path.join(meta_params['mat_path'], file+'.mat')], **meta_params)
    dataloader = DataLoader(sdf_dataset, shuffle=True, collate_fn=sdf_dataset.collate_fn, batch_size=meta_params['batch_size'], num_workers=0, drop_last=True)

    # shape embedding
    latent = training_loop.train(model=model, train_dataloader=dataloader, model_dir=save_path, is_train=False, **meta_params)

    impl_scene.add_drawable(model_name, model, latent, Mtrans=[0., 0., 1.3 * (displace // 2) * ((-1) ** (displace % 2))])

    displace += 1

impl_scene.add_point_light(nrender.PLight([0., -3., 0.], [200, 50, 50]))
impl_scene.add_point_light(nrender.PLight([0., 3., 0.], [50, 255, 50]))

nrender.render_tour_video(
    os.path.join(root_path, f'blend_{model_name}.mp4'),
    impl_scene,
    FPS=10,
    resol=1920,
    frames=90,
    view_radius=(displace // 2)+1.2
)

# calculate chamfer distance for each subject
# chamfer_dist = []
# for file in all_names:
#     recon_name = os.path.join(root_path,file,'checkpoints','test.ply')
#     gt_name = os.path.join(meta_params['point_cloud_path'],file+'.mat')
#     cd = compute_recon_error(recon_name,gt_name)
#     print(file,'\tcd:%f'%cd)
#     chamfer_dist.append(cd)

# print('Average Chamfer Distance:', np.mean(chamfer_dist))

