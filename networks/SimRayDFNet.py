# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
from . import modules
from .loss import *
from torch.nn.functional import normalize
import torchgeometry as tgm


class SimRayDistanceField(nn.Module):
    def __init__(self, model_type='sine', hidden_num=128, **kwargs):
        super().__init__()

        self.res1 = modules.RayBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=6, in_features=6, out_features=hidden_num)

        self.res2 = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=2, in_features=6+hidden_num, out_features=1)

    def inference(self, coords, dirs, *args, **kwargs):
        with torch.no_grad():
            output1 = self.res1({'coords': coords, 'dirs': dirs})['model_out']
            output2 = self.res2({'inputs': torch.cat([output1, coords, dirs], 2)})
            return torch.clamp(output2['model_out'], 0.0, 2.0)

    # input: N x (L+3)
    def forward(self, model_input, gt, *args, **kwargs):
        output1 = self.res1(model_input)['model_out']
        depth = self.res2({'inputs': torch.cat([output1, model_input['coords'], model_input['dirs']], 2)})['model_out']

        losses = sim_rdf_loss({'model_out': depth}, gt)
        
        return losses
