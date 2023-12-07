# -*- coding: utf-8 -*-
'''Training losses for ODF-Net and RayDF-Net.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def raydf_loss(model_output, gt, loss_grad_deform=5):

    gt_depth = gt['depth']

    pred_depth = model_output['model_out']

    embeddings = model_output['latent_vec']

    gradient_deform = model_output['grad_deform']

    depth_constraint = torch.clamp(pred_depth, 0.0, 1.0)-torch.clamp(gt_depth, 0.0, 1.0)

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
        'grad_deform_constraint':grad_deform_constraint.mean()* loss_grad_deform,
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