
### accelerate

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import yaml
import numpy as np

import trimesh
import torch
import configargparse
# from calculate_chamfer_distance import compute_recon_error
from networks.SimRayDFNet_release import SimRayDistanceField
from networks.SDFNet import SDFNet

import render.neural_render as nrender
import utils
from metrics import chamfer, emd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

p = configargparse.ArgumentParser()
p.add_argument('--config', required=True, help='Evaluation configuration')

# load configs
opt = p.parse_args()
with open(os.path.join(opt.config),'r') as stream:
    meta_params = yaml.safe_load(stream)
    
# define Net
if meta_params['net'] == 'sim_RDF':
    model = SimRayDistanceField(**meta_params)
else:
    model = SDFNet(**meta_params)

model.load_state_dict(torch.load(meta_params['checkpoint_path']))

# The network should be fixed for evaluation.
for param in model.res1.parameters():
    param.requires_grad = False
for param in model.res2.parameters():
    param.requires_grad = False

model.cuda()
    

import tensorrt
import torch_tensorrt
from torchvision import models
import time

def benchmark(model, inp_coords, inp_dirs, dtype='fp32', nwarmup=50, nruns=3000):
    model.eval()
    if dtype=='fp16':
        inputs = inputs.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            outputs = model(inp_coords, inp_dirs)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            outputs  = model(inp_coords, inp_dirs)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%1000==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))
    print(outputs[0][:8], outputs[0].max())

radius = 1.3
resol = 512
locs = [
    np.array([radius, 0.5, 0.2]),
    np.array([0., -0.5, radius])
]

rays = utils.get_pinhole_rays(locs[0], -locs[0], resol)
input_shape = rays.shape
inp_coords = torch.from_numpy(rays[:, :3]).reshape((1, rays.shape[0], 3)).cuda().float()
inp_dirs = torch.from_numpy(rays[:, 3:]).reshape((1, rays.shape[0], 3)).cuda().float()

# ts_model = torch.jit.script(model)

trt_model = torch_tensorrt.compile(
    model, 
    inputs=[
        torch_tensorrt.Input(inp_coords.shape, dtype=torch.float32),
        torch_tensorrt.Input(inp_dirs.shape, dtype=torch.float32)
    ],
    enabled_precisions = {torch.float32},
    workspace_size = 1 << 22
)
print('Convert over.')
#torch.jit.save(trt_model, 'trt_model.pt')
#trt_model = torch.jit.load('trt_model.pt')

# 3 check speedup
inputs = torch.randn(input_shape).to('cuda')
benchmark(model, inputs, dtype='fp32')
# benchmark(ts_model, inputs, dtype='fp32')
benchmark(trt_model, inputs, dtype='fp32')