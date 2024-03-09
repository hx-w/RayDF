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
        
huber_fn = HuberFunc(reduction="mean")
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

def dir_to_cdir(dir):
    return torch.cat([
        (torch.sin(dir[:, :, 1]) * torch.cos(dir[:, :, 0])).reshape(-1, 1),
        (torch.sin(dir[:, :, 1]) * torch.sin(dir[:, :, 0])).reshape(-1, 1),
        torch.cos(dir[:, :, 1]).reshape(-1, 1)
    ], dim=1)

def raydf_loss(model_output, gt, loss_grad_deform=5):
    gt_depth = gt['depth']

    pred_depth = torch.clamp(model_output['model_out'], 0.0, 2.0)
    embeddings = model_output['latent_vec']
    gradient_deform = model_output['grad_deform']
    coord_deform = model_output['coord_deform'].reshape(-1, 3)
    dirs = model_output['dir_old'].reshape(-1, 3)
    new_dirs = model_output['dir_new'].reshape(-1, 3)
    # deform_dir = model_output['dir_deform'].reshape(-1, 3)

    # depth prior
    depth_constraint = pred_depth - gt_depth

    # deform point-wise prior
    deform_coord_constraint = huber_fn(torch.norm(coord_deform, dim=-1), delta=0.25)
    
    # deform dirs
    # cdir_old, cdir_new = dir_to_cdir(dirs), dir_to_cdir(new_dirs)
    
    deform_dir_constraint = 1- F.cosine_similarity(dirs, new_dirs, dim=-1)
    # deform_dir_constraint += huber_fn(torch.norm(deform_dir, dim=-1), delta=0.25)


    # binary cross entropy loss
    bin_gt = torch.where(gt_depth >= 2.0, 0.0, 1.0)
    bin_pred = torch.where(pred_depth >= 2.0, 0.0, 1.0)

    cross_entropy_constraint = criterion(bin_gt, bin_pred)

    # normal_constraint = torch.where(
    #     gt_sdf == 0,
    #     1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
    #     torch.zeros_like(gradient_sdf[..., :1])
    # )

    # deformation smoothness prior
    grad_deform_constraint = gradient_deform.norm(dim=-1)

    # latent code prior
    embeddings_constraint = torch.mean(embeddings ** 2)
    
    ## inner
    inner_constraint = torch.where(gt_depth == 2, torch.zeros_like(pred_depth), torch.exp(-1e1 * torch.abs(2 - pred_depth)))

    # -----------------
    return {
        'depth': torch.abs(depth_constraint ** 1).mean() * 2e5, 
        'embeddings_constraint': embeddings_constraint.mean() * 1e2,
        'deform_coord_constraint': deform_coord_constraint.mean() * 5e2,
        'deform_dir_constraint': deform_dir_constraint.mean() * 5e2,
        'grad_deform_constraint':grad_deform_constraint.mean()* loss_grad_deform,
        'cross_entropy_constraint': cross_entropy_constraint.mean() * 2e2,
        'inner_constraint': inner_constraint.mean() * 2e2,
    }

def odf_loss(model_output, gt):

    gt_depth = gt['depth']

    pred_depth = torch.clamp(model_output['model_out'], 0.0, 2.0)
    embeddings = model_output['latent_vec']

    # depth prior
    depth_constraint = pred_depth - gt_depth

    # binary cross entropy loss
    bin_gt = torch.where(gt_depth == 2.0, 0.0, 1.0)
    bin_pred = torch.where(pred_depth == 2.0, 0.0, 1.0)

    cross_entropy_constraint = criterion(bin_gt, bin_pred)
    
    # normal_constraint = torch.where(
    #     gt_sdf == 0,
    #     1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
    #     torch.zeros_like(gradient_sdf[..., :1])
    # )

    # latent code prior
    embeddings_constraint = torch.mean(embeddings ** 2)
    ## inner
    inner_constraint = torch.where(gt_depth == 2, torch.zeros_like(pred_depth), torch.exp(-1e1 * torch.abs(2 - pred_depth)))

    # -----------------
    return {
        'depth': torch.abs(depth_constraint ** 1).mean() * 2e5,
        # 'cross_entropy_constraint': cross_entropy_constraint * 1e2,
        'embeddings_constraint': embeddings_constraint.mean() * 1e2,
        'inner_constraint': inner_constraint.mean() * 1e2
    }

def embedding_loss(model_output, gt):

    gt_depth = gt['depth']

    pred_depth = model_output['model_out']
    embeddings = model_output['latent_vec']

    # sdf regression loss from Sitzmannn et al. 2020
    depth_constraint = torch.clamp(pred_depth, 0.0, 2.0) - torch.clamp(gt_depth, 0.0, 2.0)
    embeddings_constraint = torch.mean(embeddings ** 2)
    bin_gt = torch.where(gt_depth >= 2.0, 0.0, 1.0)
    bin_pred = torch.where(pred_depth >= 2.0, 0.0, 1.0)

    cross_entropy_constraint = criterion(bin_gt, bin_pred)
    ## inner
    inner_constraint = torch.where(gt_depth == 2, torch.zeros_like(pred_depth), torch.exp(-1e1 * torch.abs(2 - pred_depth)))

    # -----------------
    return {
        'depth_constraint': torch.abs(depth_constraint ** 1).mean() * 2e5,
        'embeddings_constraint': embeddings_constraint.mean() * 2e2,
        # 'cross_entropy_constraint': cross_entropy_constraint.mean() * 2e2,
        'inner_constraint': inner_constraint.mean() * 2e2
    }

def sim_rdf_loss(model_output, gt):

    gt_depth = gt['depth']

    pred_depth = torch.clamp(model_output['model_out'], 0.0, 2.0)
    grad_depth = model_output['grad']

    # depth prior
    depth_constraint = pred_depth - gt_depth

    # binary cross entropy loss
    # bin_gt = torch.where(gt_depth == 2.0, 0.0, 1.0)
    # bin_pred = torch.where(pred_depth == 2.0, 0.0, 1.0)

    # cross_entropy_constraint = criterion(bin_gt, bin_pred)
    grad_constraint = torch.abs(grad_depth.norm(dim=-1) - 1)

    ## inner
    inner_constraint = torch.where(gt_depth == 2, torch.zeros_like(pred_depth), torch.exp(-1e2 * torch.abs(2 - pred_depth)))

    # -----------------
    return {
        'depth': torch.abs(depth_constraint ** 1).mean() * 2e5,
        # 'cross_entropy_constraint': cross_entropy_constraint * 2e2,
        'inner_constraint': inner_constraint.mean() * 2e2,
        'grad_constraint': grad_constraint.mean() * 2e2,
    }

def sim_sdf_loss(model_output, gt):

    gt_sdf = gt['sdf']
    pred_sdf = model_output['sdf']
    grad_sdf = model_output['grad']
    
    sdf_constraint = torch.clamp(pred_sdf,-1.0,1.0)-torch.clamp(gt_sdf,-1.0,1.0)
    grad_constraint = torch.abs(grad_sdf.norm(dim=-1) - 1)
    inter_constraint = torch.where(gt_sdf == 0., torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    
    return {
        'sdf': torch.abs(sdf_constraint ** 1).mean() * 2e5, 
        'inter': inter_constraint.mean() * 2e2,
        'grad_constraint': grad_constraint.mean() * 2e2,
    }
