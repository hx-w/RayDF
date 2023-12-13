#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from networks.ODFRawNet import ODFDecoder
import datasets_odfraw

import open3d as o3d
import open3d.core as o3c


def get_pinhole_rays(cam_pos: np.array, cam_dir: np.array, resol: int):
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=cam_pos + cam_dir,
        eye=cam_pos,
        up=[0, 1, 0],
        width_px=resol,
        height_px=resol,
    )
    # normalize rays
    rays = rays.numpy().reshape((-1, 6))
    rays[:, 3:] /= np.linalg.norm(rays[:, 3:], axis=1)[:, np.newaxis]
    return rays

crit = torch.nn.BCEWithLogitsLoss()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_handler = logging.StreamHandler()
formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sample,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=1e-2,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every, iter
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        # lr = initial_lr * (1 / decreased_by)
        if iter % 100 == 0:
            logging.info("iter: " + str(iter))
            logging.info("lr: " + str(lr))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in tqdm(range(num_iterations), desc='iter'):

        decoder.eval()
        data = datasets_odfraw.unpack_sdf_samples_from_ram(test_sample, num_samples)
        data = torch.from_numpy(data)
        xyzd = data[:, :5].cuda()
        depth_gt = data[:, -1:].cuda()

        depth_gt = torch.clamp(depth_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every, e)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyzd], 1).cuda().to(torch.float32)

        pred_depth = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_depth = decoder(inputs)

        pred_depth = torch.clamp(pred_depth, 0.0, clamp_dist)

        loss = 5e3 * loss_l1(pred_depth, depth_gt.reshape(pred_depth.shape))
        if l2reg:
            loss += 1e-3 * torch.mean(latent.pow(2))
        
        bin_gt = torch.where(depth_gt == 1.0, 0.0, 1.0).cuda()
        bin_pred = torch.where(pred_depth == 1.0, 0.0, 1.0)

        cross_entropy_constraint = crit(bin_gt, bin_pred)
        loss += cross_entropy_constraint
        
        loss.backward()
        optimizer.step()

        if e % 100 == 0:
            logging.info(loss.item())
            logging.info(e)
            logging.info(latent.norm())
        loss_num = loss.item()

    return loss_num, latent


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained ODF decoder to reconstruct a shape given view "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        default="datasets",
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--seed",
        dest="seed",
        default=15,
        help="random seed",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        default=256,
        help="Marching cube resolution.",
    )

    args = arg_parser.parse_args()

    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    latent_size = specs["CodeLength"]

    decoder = ODFDecoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, "ModelParams", args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = f.read().split('\n')

    mat_filenames = [os.path.join(args.data_source, specs["Tag"], f + '.mat') for f in split]
    
    # random.shuffle(npz_filenames)
    mat_filenames = sorted(mat_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, "Eval", str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)


    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, "EvalCode"
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    for ii, mat in enumerate(mat_filenames):

        if "mat" not in mat:
            continue

        full_filename = mat
        mat = mat.split('/')[-1]

        logging.debug("loading {}".format(mat))

        data_sample = datasets_odfraw.read_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                latent_filename = os.path.join(
                    reconstruction_codes_dir, mat[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                latent_filename = os.path.join(
                    reconstruction_codes_dir, mat[:-4] + ".pth"
                )

            logging.info("reconstructing {}".format(mat))

            data_sample = data_sample[torch.randperm(data_sample.shape[0])]

            start = time.time()
            if True or not os.path.isfile(latent_filename):
                err, latent = reconstruct(
                    decoder,
                    int(args.iterations),
                    latent_size,
                    data_sample,
                    0.01,  # [emp_mean,emp_var],
                    1,
                    num_samples=8000,
                    lr=5e-3,
                    l2reg=True,
                )
                logging.info("reconstruct time: {}".format(time.time() - start))
                logging.info("reconstruction error: {}".format(err))
                err_sum += err
                # logging.info("current_error avg: {}".format((err_sum / (ii + 1))))
                # logging.debug(ii)

                # logging.debug("latent: {}".format(latent.detach().cpu().numpy()))
            else:
                logging.info("loading from " + latent_filename)
                latent = torch.load(latent_filename).squeeze(0)

            decoder.eval()

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    cam_pos = np.array([0.4, -0.5, 0.0])
                    cam_dir = np.array([0.2, 0.2, 0.0])
                    resol = args.resolution
                    
                    rays = get_pinhole_rays(cam_pos, cam_dir, resol)

                    pixel_num = resol * resol
                    
                    # to theta, phi
                    inp_dirs = np.zeros(shape=(pixel_num, 2))
                    inp_dirs[:, 0] = np.arctan2(rays[:, 4], rays[:, 4])
                    inp_dirs[:, 1] = np.arccos(rays[:, 5])
                    
                    inp_coords = torch.from_numpy(rays[:, :3]).reshape((pixel_num, 3)).cuda().float()
                    inp_dirs = torch.from_numpy(inp_dirs).reshape((pixel_num, 2)).cuda().float()

                    lants = latent.expand(inp_coords.shape[0], -1)
                    
                    inps = torch.cat([lants, inp_coords, inp_dirs], 1)
                    
                    pred_depth = decoder(inps)[:, :1].squeeze(1).detach().cpu().numpy().reshape((resol, resol))
                    
                    print(pred_depth)
                    
                    style = 'coolwarm'
                    # plt.figure(figsize = (50, 50))
                    htmap = sns.heatmap(pred_depth, cmap=style, cbar=False, xticklabels=False, yticklabels=False)
                    
                    filename = os.path.join(reconstruction_dir, f"{mat[:-4]}.png")
                    htmap.get_figure().savefig(filename, pad_inches=False, bbox_inches='tight')
                    plt.close()
                    
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
