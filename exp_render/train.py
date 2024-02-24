# -*- coding: utf-8 -*-
'''Training script for RayDF-Net.
'''

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import yaml
import io
import numpy as np

import torch
from torch.utils.data import DataLoader
import configargparse
from torch import nn

import dataset
from networks import loss, modules, meta_modules
from networks.SimRayDFNet import SimRayDistanceField
from networks.SDFNet import SDFNet

from exp_render import training_loop


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    p = configargparse.ArgumentParser()
    p.add_argument('--config', type=str,default='', help='training configuration.')

    p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
    p.add_argument('--experiment_name', type=str, default='default',
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

    # General training options
    p.add_argument('--batch_size', type=int, default=256, help='training batch size.')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
    p.add_argument('--epochs', type=int, default=500, help='Number of epochs to train for.')

    p.add_argument('--epochs_til_checkpoint', type=int, default=5,
                help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=100,
                help='Time interval in seconds until tensorboard summary is saved.')

    p.add_argument('--model_type', type=str, default='sine',
                help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
    p.add_argument('--mat_path', type=str, default='', help='training data path.')

    p.add_argument('--hidden_num', type=int,default=128, help='hidden layer dimension of deform-net.')

    p.add_argument('--batch_max', type=int, default=4000, help='number of samples for each iteration.')

    # load configs if exist
    opt = p.parse_args()
    if opt.config == '':
        meta_params = vars(opt)
    else:
        with open(opt.config, 'r') as stream:
            meta_params = yaml.safe_load(stream)
    

    if meta_params['net'] == 'sim_RDF':
        model = SimRayDistanceField(**meta_params)
        df_dataset = dataset.SimRayDepthDataset(mat_path=meta_params['mat_path'], batch_max=meta_params['batch_max'])
    elif meta_params['net'] == 'sim_SDF':
        model = SDFNet(**meta_params)
        df_dataset = dataset.SimSDFDataset(mat_path=meta_params['mat_path'], batch_max=meta_params['batch_max'])

    dataloader = DataLoader(df_dataset, shuffle=True, batch_size=meta_params['batch_size'], num_workers=8, drop_last=False)

    if 'checkpoint_path' in meta_params and len(meta_params['checkpoint_path']) > 0 and os.path.isfile(meta_params['checkpoint_path']):
        state_dict=torch.load(meta_params['checkpoint_path'])
        filtered_state_dict={k:v for k,v in state_dict.items() if k.find('detach')==-1}
        model.load_state_dict(filtered_state_dict)
        print('load %s'%meta_params['checkpoint_path'])

    model = nn.DataParallel(model)
    model.cuda()

    # create save path
    root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name'])
    os.makedirs(root_path, exist_ok=True)

    with io.open(os.path.join(root_path,'model.yml'),'w',encoding='utf8') as outfile:
        yaml.dump(meta_params, outfile, default_flow_style=False, allow_unicode=True)

    # main training loop
    training_loop.train(model=model, train_dataloader=dataloader, model_dir=root_path, **meta_params)