# -*- coding: utf-8 -*-

'''Define RayDF-Net, TODO
'''

import torch
from torch import nn
from . import modules
from .meta_modules import HyperNetwork
from .loss import *


class RayDistanceField(nn.Module):
    def __init__(self, num_instances, latent_dim=128, model_type='sine', hyper_hidden_layers=1, hyper_hidden_features=256, hidden_num=128, **kwargs):
        super().__init__()
        #　We use auto-decoder framework following Park et al. 2019 (DeepSDF),
        # therefore, the model consists of latent codes for each subjects and DIF-Net decoder.

        # latent code embedding for training subjects
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        ## RayDF-Net
        # template field
        self.template_field = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=3, in_features=5, out_features=1)
        
        # Deform-Net
        self.deform_net=modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=3, in_features=5, out_features=3)

        # Hyper-Net
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, hypo_module=self.deform_net)

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
            model_output = self.deform_net(model_in, params=hypo_params)

            deformation = model_output['model_out'][:, :, :]
            new_coords = coords + deformation
            model_input_temp = {'coords': new_coords, 'dirs': dirs}
            model_output_temp = self.template_field(model_input_temp)

            return model_output_temp['model_out']

    def get_template_coords(self, coords, dirs, embedding):
        with torch.no_grad():
            model_in = {'coords': coords, 'dirs': dirs}
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)
            deformation = model_output['model_out'][:, :, :]
            new_coords = coords + deformation

            return new_coords

    def get_template_field(self, coords, dirs):
        with torch.no_grad():
            model_in = {'coords': coords, 'dirs': dirs}
            model_output = self.template_field(model_in)

            return model_output['model_out']

    # for training
    def forward(self, model_input, gt, **kwargs):

        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 3 dimensional input coordinates
        dirs = model_input['dirs'] # 2 dimensional input directions

        # get network weights for Deform-net using Hyper-net 
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)

        # [deformation field, correction field]
        model_output = self.deform_net(model_input, params=hypo_params)

        deformation = model_output['model_out'][:, :, :]  # 3 dimensional deformation field
        new_coords = coords + deformation # deform into template space

        # calculate gradient of the deformation field
        x = model_output['model_in']['coords'] # input coordinates
        u = deformation[:, :, 0]
        v = deformation[:, :, 1]
        w = deformation[:, :, 2]

        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_v = torch.autograd.grad(v, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_w = torch.autograd.grad(w, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_deform = torch.stack([grad_u, grad_v, grad_w],dim=2)  # gradient of deformation wrt. input position

        model_input_temp = {'coords':new_coords, 'dirs': dirs}

        model_output_temp = self.template_field(model_input_temp)

        depth = model_output_temp['model_out'] # SDF value in template space

        model_out = {
            'coord_deform': deformation,
            'grad_deform':grad_deform,
            'model_out': depth,
            'latent_vec':embedding,
            'hypo_params':hypo_params,
        }
        losses = raydf_loss(model_out, gt, loss_grad_deform=kwargs['loss_grad_deform'])

        return losses

    # for evaluation
    def embedding(self, embed, model_input, gt):

        coords = model_input['coords'] # 3 dimensional input coordinates
        dirs = model_input['dirs'] # 2 dimensional input directions

        # get network weights for Deform-net using Hyper-net 
        hypo_params = self.hyper_net(embed)

        # [deformation field, correction field]
        model_output = self.deform_net(model_input, params=hypo_params)

        deformation = model_output['model_out'][:, :, :] # 3 dimensional deformation field
        new_coords = coords + deformation # deform into template space

        model_input_temp = {'coords':new_coords, 'dirs': dirs}

        model_output_temp = self.template_field(model_input_temp)

        depth = model_output_temp['model_out'] # SDF value in template space

        model_out = { 'model_out': depth, 'latent_vec':embed }
        losses = embedding_loss(model_out, gt)

        return losses

if __name__ == '__main__':
    raynet = RayDistanceField(10)
    