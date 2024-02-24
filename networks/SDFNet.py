#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
from .loss import *
from . import modules


class SDFNet(nn.Module):
    def __init__(self, model_type='sine', hidden_num=128, **kwargs):
        super().__init__()

        self.res1 = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=8, in_features=3, out_features=hidden_num-3)

        self.res2 = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=4, in_features=hidden_num, out_features=1)
        # self.net = modules.RayBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=8, in_features=6, out_features=1)

    def name():
        return 'SDFNet'

    def inference(self, coords, *args, **kwargs):
        with torch.no_grad():
            output1 = self.res1({'inputs': coords})['model_out']
            output2 = self.res2({'inputs': torch.cat([output1, coords], 1)})
            return torch.clamp(output2['model_out'], -1.0, 1.0)

            output = self.net({'coords': coords, 'dirs': dirs})['model_out']
            return torch.clamp(output, 0.0, 2.0)

    # input: N x (L+3)
    def forward(self, model_input, gt, *args, **kwargs):
        output1 = self.res1({'inputs': model_input['coords']})['model_out']
        depth = self.res2({'inputs': torch.cat([output1, model_input['coords']], 2)})['model_out']
        # depth = self.net(model_input)['model_out']

        grad_sdf = torch.autograd.grad(depth, [model_input['coords']], grad_outputs=torch.ones_like(depth), create_graph=True)[0]

        losses = sim_sdf_loss({'sdf': depth, 'grad': grad_sdf}, gt)
        
        return losses
