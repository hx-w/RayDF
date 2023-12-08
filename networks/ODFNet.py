# -*- coding: utf-8 -*-

'''Define ODF-Net, TODO
'''

import torch
from torch import nn
from . import modules
from .meta_modules import HyperNetwork
from .loss import *


class OmniDistanceField(nn.Module):
    def __init__(self, num_instances, latent_dim=128, model_type='sine', hyper_hidden_layers=1, hyper_hidden_features=256, hidden_num=128, **kwargs):
        super().__init__()
        #　We use auto-decoder framework following Park et al. 2019 (DeepSDF),
        # therefore, the model consists of latent codes for each subjects and DIF-Net decoder.

        # latent code embedding for training subjects
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        ## ODF-Net
        
        self.forward_net_1 = modules.RayBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=2, in_features=5, out_features=hidden_num)
        self.forward_net_2 = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=6, in_features=5+hidden_num, out_features=1)
        
        # Hyper-Net
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, hypo_module=self.forward_net_1)

        print(self)

    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self,instance_idx):
        embedding = self.latent_codes(instance_idx)

        return embedding

    # for generation
    def inference(self, coords, dirs, embedding):
        with torch.no_grad():
            model_in = {'coords': coords, 'dirs': dirs}
            hypo_params = self.hyper_net(embedding)
            
            model_output = self.forward_net_1(model_in, params=hypo_params)
            model_2_input = torch.cat([model_output['model_out'], coords, dirs], 2)

            model_2_output = self.forward_net_2({'inputs': model_2_input})
            return model_2_output['model_out']

    # for training
    def forward(self, model_input, gt, **kwargs):

        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 3 dimensional input coordinates
        dirs = model_input['dirs'] # 2 dimensional input directions

        # get network weights for Deform-net using Hyper-net 
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)

        model_output = self.forward_net_1(model_input, params=hypo_params)
        model_2_input = torch.cat([model_output['model_out'], coords, dirs], 2)

        model_2_output = self.forward_net_2({'inputs': model_2_input})
        depth = model_2_output['model_out']

        model_out = {
            'model_out': depth,
            'latent_vec':embedding,
            'hypo_params':hypo_params,
        }
        losses = odf_loss(model_out, gt)

        return losses

    # for evaluation
    def embedding(self, embed, model_input, gt):

        coords = model_input['coords'] # 3 dimensional input coordinates
        dirs = model_input['dirs'] # 2 dimensional input directions

        # get network weights for Deform-net using Hyper-net 
        hypo_params = self.hyper_net(embed)

        model_output = self.forward_net_1(model_input, params=hypo_params)

        model_2_input = torch.cat([model_output['model_out'], coords, dirs], 2)

        model_2_output = self.forward_net_2({'inputs': model_2_input})
        depth = model_2_output['model_out']

        model_out = { 'model_out': depth, 'latent_vec':embed }
        losses = embedding_loss(model_out, gt)

        return losses

if __name__ == '__main__':
    raynet = OmniDistanceField(10)
    