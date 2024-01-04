# -*- coding: utf-8 -*-

'''Define RayDF-Net, TODO
'''

import torch
from torch import nn
from . import modules
from .meta_modules import HyperNetwork
from .loss import *
from torch.nn.functional import normalize
import torchgeometry as tgm


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
        self.template_field = modules.RayBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=5, in_features=6, out_features=1)
        
        # Deform-Net
        self.deform_net=modules.RayBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num, num_hidden_layers=3, in_features=6, out_features=6)

        # Hyper-Net
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, hypo_module=self.deform_net)

        # print(self)

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

            deform_coords = model_output['model_out'][:, :, :3]
            deform_dirs = model_output['model_out'][:, :, 3:]
            new_coords = coords + deform_coords
            
            _s = list(deform_dirs.shape[:2]) + [4, 4]
            rot = tgm.angle_axis_to_rotation_matrix(deform_dirs.reshape(-1, 3))
            rot = rot.reshape(_s)[:, :, :3, :3] # 不用齐次矩阵

            new_dirs = torch.matmul(rot, dirs.reshape(list(dirs.shape) + [1])).reshape(coords.shape)
            # new_dirs = dirs + deform_dirs
            
            # new_dirs = normalize(new_dirs, dim=2)
            
            ## modify [phi, theta]
            # new_dirs[:, :, 0] %= 2 * torch.pi
            # new_dirs[:, :, 1] = torch.clamp(new_dirs[:, :, 1], 0.0, torch.pi)
            
            model_input_temp = {'coords': new_coords, 'dirs': new_dirs}
            model_output_temp = self.template_field(model_input_temp)

            return torch.clamp(model_output_temp['model_out'], 0.0, 2.0)

    def get_template_coords_dirs(self, coords, dirs, embedding):
        with torch.no_grad():
            model_in = {'coords': coords, 'dirs': dirs}
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)
            deform_coords = model_output['model_out'][:, :, :3]
            deform_dirs = model_output['model_out'][:, :, 3:]
            
            return deform_coords, deform_dirs
            
            new_coords = coords + deform_coords
            _s = list(deform_dirs.shape[:2]) + [4, 4]
            rot = tgm.angle_axis_to_rotation_matrix(deform_dirs.reshape(-1, 3))
            rot = rot.reshape(_s)[:, :, :3, :3] # 不用齐次矩阵

            new_dirs = torch.matmul(rot, dirs.reshape(list(dirs.shape) + [1])).reshape(coords.shape)
            # new_dirs = dirs + deform_dirs

            # new_dirs = normalize(new_dirs, dim=2)

            return new_coords, new_dirs

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

        deform_coords = model_output['model_out'][:, :, :3]
        deform_dirs = model_output['model_out'][:, :, 3:]

        ## 不越界 phi
        # indx = (deform_dirs + dirs)[:, :, 1] > torch.pi
        # indxx = (deform_dirs + dirs)[:, :, 1] < 0
        
        # deform_dirs[indx][:, 1] = torch.pi - dirs[indx][:, 1]
        # deform_dirs[indxx][:, 1] = 0 - dirs[indxx][:, 1]
        ## 

        new_coords = coords + deform_coords # deform into template space        

        _s = list(deform_dirs.shape[:2]) + [4, 4]
        rot = tgm.angle_axis_to_rotation_matrix(deform_dirs.reshape(-1, 3))
        rot = rot.reshape(_s)[:, :, :3, :3] # 不用齐次矩阵

        new_dirs = torch.matmul(rot, dirs.reshape(list(dirs.shape) + [1])).reshape(coords.shape)
        # new_dirs = dirs + deform_dirs
        
        # new_dirs = normalize(new_dirs, dim=2)
        ## modify [phi, theta]
        # new_dirs[:, :, 0] %= 2 * torch.pi

        # calculate gradient of the deformation field
        x = model_output['model_in']['coords'] # input coordinates
        u = deform_coords[:, :, 0]
        v = deform_coords[:, :, 1]
        w = deform_coords[:, :, 2]

        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_v = torch.autograd.grad(v, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_w = torch.autograd.grad(w, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_deform = torch.stack([grad_u, grad_v, grad_w],dim=2)  # gradient of deformation wrt. input position

        model_input_temp = {'coords':new_coords, 'dirs': new_dirs}

        model_output_temp = self.template_field(model_input_temp)

        depth = torch.clamp(model_output_temp['model_out'], 0.0, 2.0)
        
        model_out = {
            'coord_deform': deform_coords,
            'dir_old': dirs,
            'dir_new': new_dirs,
            # 'dir_deform': deform_dirs,
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

        deform_coords = model_output['model_out'][:, :, :3]
        deform_dirs = model_output['model_out'][:, :, 3:]
 
        # ## 不越界 phi
        # indx = (deform_dirs + dirs)[:, :, 1] > torch.pi
        # indxx = (deform_dirs + dirs)[:, :, 1] < 0
        
        # deform_dirs[indx][:, 1] = torch.pi - dirs[indx][:, 1]
        # deform_dirs[indxx][:, 1] = 0 - dirs[indxx][:, 1]
        ## 
 
        new_coords = coords + deform_coords # deform into template space
        _s = list(deform_dirs.shape[:2]) + [4, 4]
        rot = tgm.angle_axis_to_rotation_matrix(deform_dirs.reshape(-1, 3))
        rot = rot.reshape(_s)[:, :, :3, :3] # 不用齐次矩阵

        new_dirs = torch.matmul(rot, dirs.reshape(list(dirs.shape) + [1])).reshape(coords.shape)
        # new_dirs = dirs + deform_dirs

        # new_dirs = normalize(new_dirs, dim=2)

        ## modify [phi, theta]
        # new_dirs[:, :, 0] %= 2 * torch.pi

        model_input_temp = {'coords':new_coords, 'dirs': new_dirs}

        model_output_temp = self.template_field(model_input_temp)

        depth = torch.clamp(model_output_temp['model_out'], 0.0, 2.0)

        model_out = { 'model_out': depth, 'latent_vec':embed }
        losses = embedding_loss(model_out, gt)

        return losses

if __name__ == '__main__':
    raynet = RayDistanceField(10)
    