# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import open3d.core as o3c
import open3d as o3d
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from tqdm import tqdm
import preprocess as prep


def get_equidistant_camera_angles(count):
    # increment = math.pi * (3 - math.sqrt(5))
    increment = (2. * np.pi) / count

    for i in range(count):
        phi = np.arcsin(-1 + 2 * i / (count - 1))
        # phi = 0.
        theta = ((i + 1) * increment) % (2 * np.pi)
        
        yield np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])   
        # yield theta, phi

def get_pinhole_rays(cam_pos: np.array, cam_dir: np.array, resol: int):
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=cam_pos + cam_dir,
        eye=cam_pos,
        up=[0, 1, 0] if (cam_dir[0] != 0 or cam_dir[2] != 0) else [0, 0, 1],
        width_px=resol,
        height_px=resol,
    )
    # normalize rays
    rays = rays.numpy().reshape((-1, 6))
    rays[:, 3:] /= np.linalg.norm(rays[:, 3:], axis=1)[:, np.newaxis]
    return rays

def generate_inference_by_rays(rays: np.array, model, embedding=None):
    inp_coords = torch.from_numpy(rays[:, :3]).reshape((1, rays.shape[0], 3)).cuda().float()
    inp_dirs = torch.from_numpy(rays[:, 3:]).reshape((1, rays.shape[0], 3)).cuda().float()
    
    if embedding is not None:
        depth = (
            model.inference(inp_coords, inp_dirs, embedding)
            .squeeze(1).detach().cpu().numpy().reshape((-1, 1))
        )
    else:
        depth = (
            model.get_template_field(inp_coords, inp_dirs)
            .squeeze(1).detach().cpu().numpy().reshape((-1, 1))
        )
    
    depth[depth[:, 0:] > 1.9] = 2.
    
    return np.concatenate([rays, depth], axis=1)
    

def generate_inference(cam_pos: np.array, cam_dir: np.array, model, resol: int, embedding=None):
    rays = get_pinhole_rays(cam_pos, cam_dir, resol) # (n, 6)
    
    return generate_inference_by_rays(rays, model, embedding)
    

def generate_scan(cam_pos: np.array, cam_dir: np.array, model, resol: int, filename: str=None, embedding=None, methods: list=['recursive', 'raw']):

    rays = get_pinhole_rays(cam_pos, cam_dir, resol) # (n, 6)
    
    # inp_coords = torch.from_numpy(rays[:, :3]).reshape((1, pixel_num, 3))
    # inp_dirs = torch.from_numpy(rays[:, 3:]).reshape((1, pixel_num, 3))
    image_arrs = []
    for mtd in methods:
        if mtd == 'recursive':
            depth = recurv_inference_by_rays(rays, model, embedding)
        elif mtd == 'raw':
            depth = generate_inference_by_rays(rays, model, embedding)[:, -1]
        else:
            raise ValueError(f'Not a valid method: {mtd}')

        depth_mat = depth.reshape((resol, resol))
        
        style = 'viridis'
        # plt.figure(figsize = (50, 50))
        htmap = sns.heatmap(depth_mat, cmap=style, cbar=False, xticklabels=False, yticklabels=False)
        
        if filename is not None:
            htmap.get_figure().savefig(filename.replace('.png', f'_{mtd}.png'), pad_inches=False, bbox_inches='tight')
            # continue

        norm = Normalize()
        cmap = cm.get_cmap(style)
        image_arrs.append(cmap(norm(depth_mat))[:, :, :3])
        
        plt.close()
    
    
    image_array = np.concatenate(image_arrs, axis=1)
    
    return image_array

def recurv_inference_by_rays(rays: np.array, model, embedding=None, thred: float=.3, stack_depth: int=0):
    depth = np.zeros(shape=(rays.shape[0], 1))
    
    samples = generate_inference_by_rays(rays, model, embedding)
    
    depth[samples[:, -1] >= 2.] = 2.
    depth[samples[:, -1] <= thred] = samples[samples[:, -1] <= thred][:, -1:]
    
    recurv_ind = np.logical_and(samples[:, -1] < 2., samples[:, -1] > thred)
    depth[recurv_ind] = thred
    
    if samples[recurv_ind].shape[0] > 0:
        sub_rays = rays[recurv_ind]
        sub_rays[:, :3] += thred * sub_rays[:, 3:]
        # print(f'recurv: {stack_depth} | {rays.shape[0]} => {sub_rays.shape[0]} with min {samples[recurv_ind][:, -1].min()}')
        depth[recurv_ind] += recurv_inference_by_rays(sub_rays, model, embedding, thred, stack_depth=stack_depth+1)
    
    depth[depth[:, 0] > 2.] = 2.

    return depth

'''
compute the first intersect point with ray and sphere,
- return [np.inf] if no intersects
- return [t]      norm(rays[:3] + t * rays[3:] - c) = r

@param: [ray] np.array with shape=(6,)
@param: [c]   np.array with shape=(3,) ball center
@param: [r]   float    ball radius

https://zhuanlan.zhihu.com/p/574032163
'''
@nb.njit
def _sphere_intersect(ray: np.array, c: np.array, r: float) -> float:
    vec_ori2c = ray[:3] - c
    
    # solve aX^2+bX+c=0
    a    = np.dot(ray[3:], ray[3:])   # a == 1, if ray's dir is normalized
    b    = 2 * np.dot(ray[3:], vec_ori2c)
    c    = np.dot(vec_ori2c, vec_ori2c) - r * r
    disc = b * b - 4 * a * c
    
    if disc <= 0:
        return np.inf

    dist_sqrt = np.sqrt(disc)
    
    q = (-b - dist_sqrt) / 2.0 if b < 0 else (-b + dist_sqrt) / 2.0
    
    t0, t1 = q / a, c / q

    t0, t1 = min(t0, t1), max(t0, t1)

    if t1 >= 0:
        return t1 if t0 < 0 else t0

    return np.inf

'''
filter rays with a sphere,

- return [ray_oris] np.array with shape=(n, 3) new origins of rays
- return [Ts]       np.array with shape=(n, 1)

@param [rays] np.array with shape=(n, 6)
@param [c]    np.array with shape=(3,) center of the sphere
@param [r]    float    radius of the sphere
'''
# @nb.guvectorize(
#     'float64[:, :], float64[:], float64, float64[:, :]',
#     '(n, m), (k), () -> (n, m)'
# )
def filter_rays_with_sphere(rays: np.array, c: np.array=np.zeros(shape=(3,), dtype=np.float32), r: float=1.3) -> np.array:
    Ts = [_sphere_intersect(ray, c, r) for ray in rays]
    
    Ts = np.array(Ts)
    
    ray_oris = rays[:, :3]
    
    val_ind = Ts < np.inf
    ray_oris[val_ind, :] += Ts[val_ind].reshape(-1, 1) * rays[val_ind, 3:]
    
    return np.concatenate([ray_oris, Ts.reshape(-1, 1)], axis=1)

def generate_scan_super(cam_pos: np.array, cam_dir: np.array, model, resol: int, filename: str=None, embedding=None, methods: list=['recursive', 'raw']):
    rays = get_pinhole_rays(cam_pos, cam_dir, resol) # (n, 6)
    
    ray_ts = filter_rays_with_sphere(rays, r=1.2)
    rays[:, :3] = ray_ts[:, :3]
    
    # inp_coords = torch.from_numpy(rays[:, :3]).reshape((1, pixel_num, 3))
    # inp_dirs = torch.from_numpy(rays[:, 3:]).reshape((1, pixel_num, 3))
    image_arrs = []
    normal_arrs = []
    for mtd in methods:
        if mtd == 'recursive':
            depth = recurv_inference_by_rays(rays, model, embedding)
        elif mtd == 'raw':
            depth = generate_inference_by_rays(rays, model, embedding)[:, -1:]
        else:
            raise ValueError(f'Not a valid method: {mtd}')

        depth[depth >= 2.] = np.inf
        depth = depth.reshape(-1, 1)
        depth = ray_ts[:, -1:] + depth

        depth_mat = depth.reshape((resol, resol))
        depth_mat[depth_mat == np.inf] = 0.
        
        style = 'gray_r'
        # plt.figure(figsize = (50, 50))
        # htmap = sns.heatmap(depth_mat, cmap=style, cbar=False, xticklabels=False, yticklabels=False)
        
        norm = Normalize()
        cmap = cm.get_cmap(style)
        image  = cmap(norm(depth_mat))[:, :, :3]

        normal_image, normal_raw = depth2normal(depth_mat)
        normal_image[depth_mat == np.inf] = np.array([255., 255., 255.], dtype=np.uint8)
        
        if filename is not None:
            image = (image * 255).astype(np.uint8)
            cv2.imwrite(filename.replace('.png', f'_{mtd}.png'), image)
            # continue
            cv2.imwrite(filename.replace(".png", f'_{mtd}_normal.png'), normal_image)
        
        plt.close()
    
        image_arrs.append(image)
        normal_arrs.append(normal_image)

    
    image_array = np.concatenate(image_arrs, axis=1)
    
    return image_array
    
def generate_tour_video_super(model, radius: float, FPS: int, frames: int, resol: int, filename: str, embedding=None, methods: list=['recursive', 'raw']):
    for mtd in methods:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filename.replace('.mp4', f'_{mtd}.mp4'), fourcc, FPS, (resol, resol), True)
        video_writer_normal = cv2.VideoWriter(filename.replace('.mp4', f'_{mtd}_normal.mp4'), fourcc, FPS, (resol, resol), True)
        
        for cam_pos in tqdm(get_equidistant_camera_angles(frames), desc='gen video', total=frames):
            cam_pos = radius * cam_pos
            rays = get_pinhole_rays(cam_pos, -cam_pos, resol)
            
            ray_ts = filter_rays_with_sphere(rays, r=1.3)
            rays[:, :3] = ray_ts[:, :3]
             
            if mtd == 'recursive':
                depth = recurv_inference_by_rays(rays, model, embedding)
            elif mtd == 'raw':
                depth = generate_inference_by_rays(rays, model, embedding)[:, -1:]
            else:
                raise ValueError(f'Not a valid method: {mtd}')

            depth[depth >= 2.] = np.inf
            depth = depth.reshape(-1, 1)
            depth = ray_ts[:, -1:] + depth
            mat = depth.reshape((resol, resol))
            
            mat[mat >= np.inf] = 0.

            style = 'gray_r'
            # cmap = sns.cubehelix_palette(start=0.0, gamma=0.8, as_cmap=True)
            
            norm = Normalize()
            cmap = cm.get_cmap(style)
            img = cmap(norm(mat))[:, :, :3]
            
            img = (img * 255.).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            normal_map, normal_raw = depth2normal(mat)
            normal_map[mat == np.inf] = np.array([255., 255., 255.], dtype=np.uint8)
            
            video_writer.write(img)
            video_writer_normal.write(normal_map)
            
        video_writer.release()
        video_writer_normal.release()

        cv2.destroyAllWindows()

def generate_pointcloud_super(model, counts: int, radius: float, embedding=None, filter_=True):
    in_ball_cam_pos_1 = prep.sample_uniform_points_in_unit_sphere(counts)
    in_ball_cam_pos_2 = prep.sample_uniform_points_in_unit_sphere(counts)
    free_dir = in_ball_cam_pos_2 - in_ball_cam_pos_1
    free_dir /= np.linalg.norm(free_dir, axis=1)[:, np.newaxis]
    free_ori = in_ball_cam_pos_1 * radius

    rays = np.concatenate([free_ori, free_dir], axis=1)

    depth = recurv_inference_by_rays(rays, model, embedding, thred=0.2, stack_depth=0)
    
    samples = np.concatenate([rays, depth], axis=1)
    
    samples = samples[samples[:, -1] < 2.]
    points = samples[:, :3] + samples[:, -1:] * samples[:, 3:-1]

    if not filter_:
        return points

    ## filter 正交投影过滤
    ### by open3d
    raw_points_num = points.shape[0]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    newpcd = pcd.select_by_index(ind)
    
    points = np.asarray(newpcd.points)
    
    new_points_num = points.shape[0]
    
    print(f'> pointcloud outlier removed: {raw_points_num - new_points_num}')

    return points

def depth2normal(depth_mat: np.array):
    depth_mat[depth_mat == np.inf] = 0.
    
    depth_mat = depth_mat.astype(np.float32)
    rows, cols = depth_mat.shape

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Calculate the partial derivatives of depth with respect to x and y
    dx = cv2.Sobel(depth_mat, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(depth_mat, cv2.CV_32F, 0, 1)

    # Compute the normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)
    model_normal = normal

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    # Save the normal map to a file
    normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)

    depth_mat[depth_mat == 0.] = np.inf
    return normal_bgr, model_normal

if __name__ == '__main__':
    rays = get_pinhole_rays(np.array([-2., 0, 0]), np.array([2, 0, 0]), 256)

    rays_Ts = filter_rays_with_sphere(rays, np.zeros(shape=(3,), dtype=np.float32), 1.3)

    print(rays_Ts)
