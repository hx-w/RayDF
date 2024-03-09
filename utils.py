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
from scipy.linalg import expm, norm
import point_cloud_utils as pcu
    
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
        up=[0, -1, 0] if (cam_dir[0] != 0 or cam_dir[2] != 0) else [0, 0, 1],
        width_px=resol,
        height_px=resol,
    )
    # normalize rays
    rays = rays.numpy().reshape((-1, 6))
    rays[:, 3:] /= np.linalg.norm(rays[:, 3:], axis=1)[:, np.newaxis]
    return rays

def generate_inference_by_points(coords: np.array, model):
    inp_coords = torch.from_numpy(coords[:, :3]).reshape((1, coords.shape[0], 3)).cuda().float()
    sdf = (
        model.inference(inp_coords) 
        .squeeze(1).detach().cpu().numpy().reshape((-1, 1))
    )
    return np.clip(sdf, -1.0, 1.0)

def generate_inference_by_rays(rays: np.array, model, embedding=None, is_sim=False):
    inp_coords = torch.from_numpy(rays[:, :3]).reshape((1, rays.shape[0], 3)).cuda().float()
    inp_dirs = torch.from_numpy(rays[:, 3:]).reshape((1, rays.shape[0], 3)).cuda().float()
    
    if embedding is not None or is_sim:
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

def generate_deformation_by_rays(rays: np.array, model, embedding):
    inp_coords = torch.from_numpy(rays[:, :3]).reshape((1, rays.shape[0], 3)).cuda().float()
    inp_dirs = torch.from_numpy(rays[:, 3:]).reshape((1, rays.shape[0], 3)).cuda().float()
    
    deform_coords, deform_dirs = model.get_template_coords_dirs(inp_coords, inp_dirs, embedding)
    deform_coords = deform_coords.squeeze(1).detach().cpu().numpy().reshape((-1, 3))
    deform_dirs   = deform_dirs.squeeze(1).detach().cpu().numpy().reshape((-1, 3))
    
    def get_dirs_norms(arr):
        norms = np.linalg.norm(arr, axis=1)
        normed_arr = arr / norms[:, np.newaxis]
        
        style = 'gray_r'
        norm = Normalize()
        cmap = cm.get_cmap(style)
        dnorm = (cmap(norm(norms))[:, :3] * 255).astype(np.uint8)
        
        ddirs = (normed_arr + 1) * 127.5
        ddirs = ddirs.clip(0, 255).astype(np.uint8)

        return ddirs, dnorm
    
    return *get_dirs_norms(deform_coords), *get_dirs_norms(deform_dirs)

def recurv_inference_by_rays(rays: np.array, model, embedding=None, thred: float=.2, stack_depth: int=0, is_sim: bool=False):
    if stack_depth > 20:
        return None
    depth = np.zeros(shape=(rays.shape[0], 1))
    
    samples = generate_inference_by_rays(rays, model, embedding, is_sim)
    
    depth[samples[:, -1] >= 2.] = 2.
    depth[samples[:, -1] <= thred] = samples[samples[:, -1] <= thred][:, -1:]
    
    recurv_ind = np.logical_and(samples[:, -1] < 2., samples[:, -1] > thred)
    depth[recurv_ind] = thred
    
    if samples[recurv_ind].shape[0] > 0:
        sub_rays = rays[recurv_ind]
        sub_rays[:, :3] += thred * sub_rays[:, 3:]
        
        # print(f'recurv: {stack_depth} | {rays.shape[0]} => {sub_rays.shape[0]} with min {samples[recurv_ind][:, -1].min()}')
        temp = recurv_inference_by_rays(sub_rays, model, embedding, thred, stack_depth=stack_depth+1, is_sim=is_sim)
        if temp is None:
            return None
        depth[recurv_ind] += temp
    
    if depth is None:
        return None

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
    Ts = np.array([_sphere_intersect(ray, c, r) for ray in rays])
    
    ray_oris = rays[:, :3]
    
    val_ind = Ts < np.inf
    ray_oris[val_ind, :] += Ts[val_ind].reshape(-1, 1) * rays[val_ind, 3:]
    
    return np.concatenate([ray_oris, Ts.reshape(-1, 1)], axis=1)

def create_sdf_slice_image(model, vol_size, img_size, x_axis, y_axis, z_axis):
    sample_pnts = None
    if x_axis is not None:
        sample_pnts = torch.Tensor([
            (x_axis, (x / img_size) * vol_size - vol_size / 2, (z / img_size) * vol_size - vol_size / 2)
            for x in range(img_size) for z in range(img_size)
        ]).cuda()
    elif y_axis is not None:
        sample_pnts = torch.Tensor([
            ((x / img_size) * vol_size - vol_size / 2, y_axis, (z / img_size) * vol_size - vol_size / 2)
            for x in range(img_size) for z in range(img_size)
        ]).cuda()
    else:
        sample_pnts = torch.Tensor([
            ((x / img_size) * vol_size - vol_size / 2, (z / img_size) * vol_size - vol_size / 2, z_axis)
            for x in range(img_size) for z in range(img_size)
        ]).cuda()

    sdfs = (
        model.inference(sample_pnts)
        .squeeze(1).detach().cpu().numpy().reshape((img_size, img_size))
    )

    style = 'coolwarm'
    plt.figure(figsize = (10, 10))
    htmap = sns.heatmap(sdfs, cmap=style, cbar=False, xticklabels=False, yticklabels=False)
    # if filename:
    #     htmap.get_figure().savefig(filename, pad_inches=False, bbox_inches='tight')
    canvas = htmap.get_figure().canvas
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(height, width, 3)
    return image_array

def generate_scan_super_sdf(cam_pos: np.array, cam_dir: np.array, model, resol: int, filename: str=None):
    rays = get_pinhole_rays(cam_pos, cam_dir, resol) # (n, 6)

    ray_ts = filter_rays_with_sphere(rays, r=1.3)
    rays[:, :3] = ray_ts[:, :3]
    
    max_step = 20
    thred = 1e-4
    
    val_inds = (ray_ts[:, 3:] < np.inf).flatten()
    march_depth = np.zeros_like(ray_ts[:, 3:])
    march_depth[val_inds] = generate_inference_by_points(rays[val_inds, :3], model)
    
    ends = np.linalg.norm(rays[val_inds, :3] + march_depth[val_inds, :] * rays[val_inds, 3:], axis=1)
    march_depth[val_inds][ends > 1.3] = -1
    
    fil_inds = (march_depth > thred).flatten()
    
    curr_step = 1
    while march_depth[fil_inds].shape[0] > 0:
        curr_step += 1
        if curr_step >= max_step:
            print('=> Ray marching iter max')
            break
        
        new_depth = np.copy(march_depth)
        new_depth[fil_inds, :] = march_depth[fil_inds, :] + generate_inference_by_points(rays[fil_inds, :3] + rays[fil_inds, 3:] * march_depth[fil_inds, :], model)
        ends = np.linalg.norm(rays[fil_inds, :3] + new_depth[fil_inds, :] * rays[fil_inds, 3:], axis=1)
        new_depth[fil_inds][ends > 1.3] = -1.
        
        fil_inds = ((new_depth - march_depth) > thred).flatten()
        march_depth = np.copy(new_depth)
    
    march_depth[march_depth < 0] = np.inf
    depth = ray_ts[:, -1:] + march_depth
    
    depth_mat = depth.reshape((resol, resol))
    depth_mat[depth_mat == np.inf] = 0.
    
    style = 'gray_r'
    
    norm = Normalize()
    cmap = cm.get_cmap(style)
    image  = cmap(norm(depth_mat))[:, :, :3]

    # depth_mat[depth_mat == 0.] = np.inf
    normal_image, _ = depth2normal_tangentspace(depth_mat)
    normal_image[depth_mat == 0.] = np.array([255., 255., 255.], dtype=np.uint8)
    
    normal_rbg = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
    normal_rbg = (normal_rbg.astype(np.float64) / 255.)
 
    return np.concatenate([image, normal_image], axis=0)
    

def generate_scan_super(cam_pos: np.array, cam_dir: np.array, model, resol: int, filename: str=None, embedding=None, methods: list=['recursive', 'raw'], is_sim=False):
    rays = get_pinhole_rays(cam_pos, cam_dir, resol) # (n, 6)
    
    ray_ts = filter_rays_with_sphere(rays, r=1.3)
    rays[:, :3] = ray_ts[:, :3]
    
    image_arrs = []
    normal_arrs = []
    
    for mtd in methods:
        if mtd == 'recursive':
            depth = recurv_inference_by_rays(rays, model, embedding, is_sim=is_sim)
        elif mtd == 'raw':
            depth = generate_inference_by_rays(rays, model, embedding, is_sim=is_sim)[:, -1:]
        else:
            raise ValueError(f'Not a valid method: {mtd}')

        if depth is None:
            continue

        depth[depth >= 2.] = np.inf
        depth = depth.reshape(-1, 1)
        depth = ray_ts[:, -1:] + depth

        depth_mat = depth.reshape((resol, resol))
        inf_mask = depth_mat == np.inf
        depth_mat[inf_mask] = np.min(depth_mat, keepdims=False, axis=None)
        
        style = 'gray_r'
        
        norm = Normalize()
        cmap = cm.get_cmap(style)
        image  = cmap(norm(depth_mat))[:, :, :3]
        image[inf_mask] = np.array([1, 1, 1])
        depth_mat[inf_mask] = 0.

        # depth_mat[depth_mat == 0.] = np.inf
        normal_image, _ = depth2normal_tangentspace(depth_mat)
        normal_image[depth_mat == 0.] = np.array([255., 255., 255.], dtype=np.uint8)
        
        if mtd == 'raw' and embedding is not None:
            deform_T, deform_Tn, deform_R, deform_Rn = generate_deformation_by_rays(rays, model, embedding)
            deform_T = deform_T.reshape((resol, resol, 3))
            deform_Tn = deform_Tn.reshape((resol, resol, 3))
            deform_R = deform_R.reshape((resol, resol, 3))
            deform_Rn = deform_Rn.reshape((resol, resol, 3))
            
            deform_T = cv2.cvtColor(deform_T, cv2.COLOR_RGB2BGR)
            deform_R = cv2.cvtColor(deform_R, cv2.COLOR_RGB2BGR)
            deform_T[depth_mat == 0.] = np.array([255, 255, 255], dtype=np.uint8)
            deform_Tn[depth_mat == 0.] = np.array([255, 255, 255], dtype=np.uint8)
            deform_R[depth_mat == 0.] = np.array([255, 255, 255], dtype=np.uint8)
            deform_Rn[depth_mat == 0.] = np.array([255, 255, 255], dtype=np.uint8)
            
            cv2.imwrite(filename.replace('.png', f'_deform_T.png'), deform_T)
            cv2.imwrite(filename.replace('.png', f'_deform_Tn.png'), deform_Tn)
            cv2.imwrite(filename.replace('.png', f'_deform_R.png'), deform_R)
            cv2.imwrite(filename.replace('.png', f'_deform_Rn.png'), deform_Rn)
        
        if filename is not None:
            image = (image * 255).astype(np.uint8)
            cv2.imwrite(filename.replace('.png', f'_{mtd}.png'), image)
            cv2.imwrite(filename.replace(".png", f'_{mtd}_normal.png'), normal_image)
            
        # plt.close()
    
        normal_rbg = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
        normal_rbg = (normal_rbg.astype(np.float64) / 255.)
        image_arrs.append(image)
        normal_arrs.append(normal_rbg)

    normal_array = np.concatenate(normal_arrs, axis=1)
    image_array = np.concatenate(image_arrs, axis=1)
    
    return np.concatenate([image_array, normal_array], axis=0)
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
            
            normal_map, _ = depth2normal_tangentspace(mat)
            normal_map[mat == np.inf] = np.array([255., 255., 255.], dtype=np.uint8)
            normal_map[mat == 0.] = np.array([255., 255., 255.], dtype=np.uint8)
            
            video_writer.write(img)
            video_writer_normal.write(normal_map)
            
        video_writer.release()
        video_writer_normal.release()

        cv2.destroyAllWindows()

def generate_pointcloud_super(model, counts: int, radius: float, embedding=None, filter_=True, is_sim=False):
    in_ball_cam_pos_1 = prep.sample_uniform_points_in_unit_sphere(counts)
    in_ball_cam_pos_2 = prep.sample_uniform_points_in_unit_sphere(counts)
    free_dir = in_ball_cam_pos_2 - in_ball_cam_pos_1
    free_dir /= np.linalg.norm(free_dir, axis=1)[:, np.newaxis]
    free_ori = in_ball_cam_pos_1 * radius

    rays = np.concatenate([free_ori, free_dir], axis=1)

    depth = recurv_inference_by_rays(rays, model, embedding, thred=0.2, stack_depth=0, is_sim=is_sim)
    
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
    
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
    # newpcd = pcd.select_by_index(ind)
    
    # points = np.asarray(newpcd.points)
    points = np.asarray(pcd.points)
    
    new_points_num = points.shape[0]
    
    print(f'> pointcloud outlier removed: {raw_points_num - new_points_num}')

    return points

def depth2normal_exp(depth_mat: np.array, rays_mat: np.array):
    resol = depth_mat.shape[0]
    # depth_mat = depth_mat.reshape((resol, resol, 1))
    inv_mask = depth_mat != np.inf
    rays_dir = rays_mat[:, :, 3:]
    
    Td = np.zeros_like(rays_dir)
    Td[inv_mask] = depth_mat[inv_mask].reshape(-1, 1) * rays_dir[inv_mask, :]

    # Calculate the partial derivatives of depth with respect to x and y
    # dx = cv2.Sobel(Td, cv2.CV_32F, 1, 0)
    # dy = cv2.Sobel(Td, cv2.CV_32F, 0, 1)
    dx = cv2.Scharr(Td, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(Td, cv2.CV_32F, 0, 1)
    
    dx_dy = np.cross(dx, dy, axis=2)
    dx_dy[depth_mat == np.inf] = np.array([0, 0, 1])
    dx_dy /= np.linalg.norm(dx_dy, axis=2)[:, :, np.newaxis]
    
    normal = np.ones_like(rays_dir) * np.array([0., 0., 1.])
    normal[inv_mask] = dx_dy[inv_mask]
    normal_dir_not_correct = (rays_dir * normal).sum(axis=-1) > 0
    normal[normal_dir_not_correct] = -normal[normal_dir_not_correct]
    
    normals = (normal * 0.5 + 0.5) * 255.0
    normals = normals.astype(np.uint8)

    # 使用 bilateralFilter 去噪
    d = 128
    sigmaColor = 75
    sigmaSpace = 75
    normals_filtered = cv2.bilateralFilter(normals, d, sigmaColor, sigmaSpace)

    # 如果我们需要进一步处理数据，可能需要把它转回 [0,1] 区间
    normals_filtered = normals_filtered.astype(np.float32) / 255.0
    normal = (normals_filtered - 0.5) / 0.5

    return normal
    

def depth2normal_tangentspace(depth_mat: np.array):
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
    # normal[:, :, :] /= np.linalg.norm(normal[:, :, :], axis=2)[:, :, np.newaxis]
    raw_normal = normal
    

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    # Save the normal map to a file
    normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)

    depth_mat[depth_mat == 0.] = np.inf
    return normal_bgr, raw_normal

def depth2normal_worldspace(rays: np.array, depth_mat: np.array):
    w, h = depth_mat.shape
    normals = np.ones(shape=(w * h, 3)) * np.array([0., 0., 1.])
    
    depth = depth_mat.flatten()
    
    pts = rays[depth < np.inf, :3] + depth[depth < np.inf].reshape(-1, 1) * rays[depth < np.inf, 3:]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=64))
    normals[depth < np.inf, :] = np.asarray(pcd.normals)
    
    # check normal direction: if ray dir and normal angle is smaller than 90, reverse normal
    # ray_dirs = rays[depth < np.inf, 3:]
    normal_dir_not_correct = (rays[:, 3:] * normals).sum(axis=-1) > 0
    normals[normal_dir_not_correct] = -normals[normal_dir_not_correct]

    return normals.reshape((w, h, 3))

def depth2normal_worldspace_super(rays, depth_mat):
    w, h = depth_mat.shape
    normals = np.ones(shape=(w * h, 3)) * np.array([0., 0., 1.])
    
    depth = depth_mat.flatten()
    
    pts = rays[depth < np.inf, :3] + depth[depth < np.inf].reshape(-1, 1) * rays[depth < np.inf, 3:]
    # normal = pcu.estimate_point_cloud_normals_ball(pts, 0.1)
    idx, normal = pcu.estimate_point_cloud_normals_knn(pts, 512)
    normals[depth < np.inf, :] = normal
    normal_dir_not_correct = (rays[:, 3:] * normals).sum(axis=-1) > 0
    normals[normal_dir_not_correct] = -normals[normal_dir_not_correct]
    
    return normals.reshape((w, h, 3))

def get_rotation_matrix_from_points(p1: np.array, p2: np.array):
    u = p1 / np.linalg.norm(p1)
    v = p2 / np.linalg.norm(p2)

    if np.linalg.norm(u - v) < 1e-6:
        return np.identity(3)

    # 计算旋转轴
    axis = np.cross(u, v)

    # 计算旋转角度
    angle = np.arccos(np.dot(u, v))

    return expm(np.cross(np.eye(3), axis/norm(axis)*angle)).T

if __name__ == '__main__':
    rays = get_pinhole_rays(np.array([-2., 0, 0]), np.array([2, 0, 0]), 256)

    rays_Ts = filter_rays_with_sphere(rays, np.zeros(shape=(3,), dtype=np.float32), 1.3)

    print(rays_Ts)
