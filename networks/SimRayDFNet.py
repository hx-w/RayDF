# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
from . import modules
from .loss import *
from torch.nn.functional import normalize
import torchgeometry as tgm
import numpy as np


class SimRayDistanceField(nn.Module):
    def __init__(self, model_type='sine', hidden_num=128, **kwargs):
        super().__init__()

        self.res1 = modules.RayBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=8, in_features=6, out_features=hidden_num-6)

        self.res2 = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=4, in_features=hidden_num, out_features=1)
        # self.net = modules.RayBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=8, in_features=6, out_features=1)

    def name():
        return 'SimRayDistanceField'

    def inference(self, coords, dirs, *args, **kwargs):
        with torch.no_grad():
            output1 = self.res1({'coords': coords, 'dirs': dirs})['model_out']
            output2 = self.res2({'inputs': torch.cat([output1, coords, dirs], 2)})
            return torch.clamp(output2['model_out'], 0.0, 2.0)

            output = self.net({'coords': coords, 'dirs': dirs})['model_out']
            return torch.clamp(output, 0.0, 2.0)

    # input: N x (L+3)
    def forward(self, model_input, gt, *args, **kwargs):
        output1 = self.res1(model_input)['model_out']
        depth = self.res2({'inputs': torch.cat([output1, model_input['coords'], model_input['dirs']], 2)})['model_out']
        # depth = self.net(model_input)['model_out']

        grad_rdf = torch.autograd.grad(depth, [model_input['dirs']], grad_outputs=torch.ones_like(depth), create_graph=True)[0]
    
        losses = sim_rdf_loss({'model_out': depth, 'grad': grad_rdf}, gt)
        
        return losses
