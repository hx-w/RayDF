# -*- coding: utf-8 -*-
'''Main training loop for RayDF-Net.
'''
import os
import time
import gc
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import numpy as np
import trimesh

from depth_visual import generate_scan, generate_inference_by_rays, recurv_inference_by_rays
import preprocess as prep

radius = 1.5
def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_schedules=None, is_train=True, **kwargs):
    print('Training Info:')
    print('data_path:\t\t',kwargs['mat_path'])
    print('num_instances:\t\t',kwargs['num_instances'])
    print('batch_size:\t\t',kwargs['batch_size'])
    print('epochs:\t\t\t',epochs)
    print('learning rate:\t\t',lr)
    for key in kwargs:
        if 'loss' in key:
            print(key+':\t',kwargs[key])
    
    if is_train:
        optim = torch.optim.Adam(lr=lr, params=model.parameters())
        if 'checkpoint_path' in kwargs and len(kwargs['checkpoint_path'])>0 and os.path.isfile(kwargs['checkpoint_path']):
            state_dict = torch.load(kwargs['checkpoint_path'].replace('model', 'optim'))
            optim.load_state_dict(state_dict)
    else:
        embedding = model.latent_codes(torch.zeros(1).long().cuda()).clone().detach() # initialization for evaluation stage
        embedding.requires_grad = True
        optim = torch.optim.Adam(lr=lr, params=[embedding])

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            # if not epoch % epochs_til_checkpoint and epoch:

            #     if is_train:
            #         torch.save(model.module.state_dict(), os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if is_train:
                    losses = model(model_input, gt, **kwargs)
                else:
                    losses = model.embedding(embedding, model_input,gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    if is_train:
                        if kwargs['net'] == 'RayDistanceField':
                            temp_slice_to_X = generate_scan(cam_pos=np.array([radius, 0.0, 0.0]), cam_dir=np.array([-radius, 0.0, 0.0]), model=model.module, resol=256)
                            temp_slice_to_Y = generate_scan(cam_pos=np.array([0.0, radius, 0.0]), cam_dir=np.array([0.0, -radius, 0.0]), model=model.module, resol=256)
                            temp_slice_to_Z = generate_scan(cam_pos=np.array([0.0, 0.0, radius]), cam_dir=np.array([0.0, 0.0, -radius]), model=model.module, resol=256)
                            writer.add_image('template_to_X', temp_slice_to_X, total_steps, dataformats='HWC')
                            writer.add_image('template_to_Y', temp_slice_to_Y, total_steps, dataformats='HWC')
                            writer.add_image('template_to_Z', temp_slice_to_Z, total_steps, dataformats='HWC')

                        torch.save(model.module.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current.pth'))
                        torch.save(optim.state_dict(),
                                    os.path.join(checkpoints_dir, 'optim_current.pth'))

                optim.zero_grad()
                train_loss.backward()
                optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                total_steps += 1
            
            gc.collect()

        if is_train:
            pass
            # torch.save(model.module.cpu().state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
        else:
            generate_scan(
                cam_pos=np.array([radius, 0.0, 0.0]),
                cam_dir=np.array([-radius, 0.0, 0.0]),
                model=model,
                resol=256,
                filename=os.path.join(checkpoints_dir, 'test_X.png'),
                embedding=embedding
            )
            generate_scan(
                cam_pos=np.array([0.0, radius, 0.0]),
                cam_dir=np.array([0.0, -radius, 0.0]),
                model=model,
                resol=256,
                filename=os.path.join(checkpoints_dir, 'test_Y.png'),
                embedding=embedding
            )
            generate_scan(
                cam_pos=np.array([0.0, 0.0, radius]),
                cam_dir=np.array([0.0, 0.0, -radius]),
                model=model,
                resol=256,
                filename=os.path.join(checkpoints_dir, 'test_Z.png'),
                embedding=embedding
            )

            # reconstruct pointcloud
            counts = 50000            
            in_ball_cam_pos_1 = prep.sample_uniform_points_in_unit_sphere(counts)
            in_ball_cam_pos_2 = prep.sample_uniform_points_in_unit_sphere(counts)
            free_dir = in_ball_cam_pos_2 - in_ball_cam_pos_1
            free_dir /= np.linalg.norm(free_dir, axis=1)[:, np.newaxis]
            free_ori = in_ball_cam_pos_1 * radius

            rays = np.concatenate([free_ori, free_dir], axis=1)

            depth = recurv_inference_by_rays(rays, model, embedding, stack_depth=0)
            
            samples = np.concatenate([rays, depth], axis=1)
            
            samples = samples[samples[:, -1] < 2.]
            points = samples[:, :3] + samples[:, -1:] * samples[:, 3:-1]

            trimesh.PointCloud(points).export(os.path.join(checkpoints_dir, 'test.ply'))
