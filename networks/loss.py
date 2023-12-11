# -*- coding: utf-8 -*-
'''Training losses for ODF-Net and RayDF-Net.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class HuberFunc(nn.Module):
    def __init__(self, reduction=None):
        super(HuberFunc, self).__init__()
        self.reduction = reduction

    def forward(self, x, delta):
        n = torch.abs(x)
        cond = n < delta
        l = torch.where(cond, 0.5 * n ** 2, n*delta - 0.5 * delta**2)
        if self.reduction is None or self.reduction == "mean":
            return torch.mean(l)
        else:
            return torch.sum(l)
        
huber_fn = HuberFunc()
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCEWithLogitsLoss()

def raydf_loss(model_output, gt, loss_grad_deform=5):

    gt_depth = gt['depth']

    pred_depth = model_output['model_out']
    embeddings = model_output['latent_vec']
    gradient_deform = model_output['grad_deform']
    coord_deform = model_output['coord_deform']
    dir_deform = model_output['dir_deform']

    # depth prior
    depth_constraint = torch.clamp(pred_depth, 0.0, 1.0)-torch.clamp(gt_depth, 0.0, 1.0)

    # deform point-wise prior
    deform_coord_constraint = huber_fn(torch.norm(coord_deform, dim=1), delta=0.75)
    deform_dir_constraint = huber_fn(torch.norm(dir_deform, dim=1), delta=0.5)
    
    # normal_constraint = torch.where(
    #     gt_sdf == 0,
    #     1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
    #     torch.zeros_like(gradient_sdf[..., :1])
    # )

    # deformation smoothness prior
    grad_deform_constraint = gradient_deform.norm(dim=-1)

    # latent code prior
    embeddings_constraint = torch.mean(embeddings ** 2)

    # -----------------
    return {
        'depth': torch.abs(depth_constraint).mean() * 3e4, 
        'embeddings_constraint': embeddings_constraint.mean() * 1e6,
        'deform_coord_constraint': deform_coord_constraint.mean() * 2e3,
        'deform_dir_constraint': deform_dir_constraint.mean() * 2e3,
        'grad_deform_constraint':grad_deform_constraint.mean()* loss_grad_deform,
    }

def odf_loss(model_output, gt):

    gt_depth = gt['depth']

    pred_depth = torch.clamp(model_output['model_out'], 0.0, 1.0)
    embeddings = model_output['latent_vec']

    # depth prior
    depth_constraint = pred_depth - gt_depth

    # binary cross entropy loss
    
    bin_gt = torch.where(gt_depth == 1.0, 0.0, 1.0)
    bin_pred = torch.where(pred_depth == 1.0, 0.0, 1.0)

    cross_entropy_constraint = criterion(bin_gt, bin_pred)
    
    # normal_constraint = torch.where(
    #     gt_sdf == 0,
    #     1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
    #     torch.zeros_like(gradient_sdf[..., :1])
    # )

    # latent code prior
    embeddings_constraint = torch.mean(embeddings ** 2)

    # -----------------
    return {
        'depth': torch.abs(depth_constraint ** 2).mean() * 5,
        'cross_entropy_constraint': cross_entropy_constraint,
        'embeddings_constraint': embeddings_constraint.mean() * 1e-3,
    }

def embedding_loss(model_output, gt):

    gt_depth = gt['depth']

    pred_depth = model_output['model_out']
    embeddings = model_output['latent_vec']

    # sdf regression loss from Sitzmannn et al. 2020
    depth_constraint = torch.clamp(pred_depth, 0.0, 1.0) - torch.clamp(gt_depth, 0.0, 1.0)
    embeddings_constraint = torch.mean(embeddings ** 2)

    # -----------------
    return {
        'sdf': torch.abs(depth_constraint).mean() * 3e4,
        'embeddings_constraint': embeddings_constraint.mean() * 1e6
    }