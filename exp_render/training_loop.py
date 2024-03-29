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

import utils

radius = 1.4
def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_schedules=None, **kwargs):
    print('Training Info:')
    print('data_path:\t\t',kwargs['mat_path'])
    print('batch_size:\t\t',kwargs['batch_size'])
    print('epochs:\t\t\t',epochs)
    print('learning rate:\t\t',lr)
    for key in kwargs:
        if 'loss' in key:
            print(key+':\t',kwargs[key])
    
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    if 'checkpoint_path' in kwargs and len(kwargs['checkpoint_path'])>0 and os.path.isfile(kwargs['checkpoint_path']):
        state_dict = torch.load(kwargs['checkpoint_path'].replace('model', 'optim'))
        optim.load_state_dict(state_dict)

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

                losses = model(model_input, gt, **kwargs)


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
                    if kwargs['net'] == 'sim_RDF':
                        temp_slice_to_X = utils.generate_scan_super(cam_pos=np.array([-radius, 0.1, 0.1]), cam_dir=-np.array([-radius, 0.1, 0.1]), model=model.module, resol=1024, is_sim=True)
                        temp_slice_to_Y = utils.generate_scan_super(cam_pos=np.array([0.1, radius, 0.1]), cam_dir=-np.array([0.1, radius, 0.1]), model=model.module, resol=1024, is_sim=True)
                        temp_slice_to_Z = utils.generate_scan_super(cam_pos=np.array([0.1, 0.1, -radius]), cam_dir=-np.array([0.1, 0.1, -radius]), model=model.module, resol=1024, is_sim=True)
                    elif kwargs['net'] == 'sim_SDF':
                        # temp_slice_to_X = utils.generate_scan_super_sdf(cam_pos=np.array([radius, 0.0, 0.0]), cam_dir=np.array([-radius, 0.0, 0.0]), model=model.module, resol=1024)
                        # temp_slice_to_Y = utils.generate_scan_super_sdf(cam_pos=np.array([0.0, radius, 0.0]), cam_dir=np.array([0.0, -radius, 0.0]), model=model.module, resol=1024)
                        # temp_slice_to_Z = utils.generate_scan_super_sdf(cam_pos=np.array([0.0, 0.0, radius]), cam_dir=np.array([0.0, 0.0, -radius]), model=model.module, resol=1024)
                        temp_slice_to_X = utils.create_sdf_slice_image(model.module, 1.5*radius, 128, None, 0, None)
                        temp_slice_to_Y = utils.create_sdf_slice_image(model.module, 1.5*radius, 128, 0, None, None)
                        temp_slice_to_Z = utils.create_sdf_slice_image(model.module, 1.5*radius, 128, None, None, 0)
                        
                    writer.add_image('preview_X', temp_slice_to_X, total_steps, dataformats='HWC')
                    writer.add_image('preview_Y', temp_slice_to_Y, total_steps, dataformats='HWC')
                    writer.add_image('preview_Z', temp_slice_to_Z, total_steps, dataformats='HWC')

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
