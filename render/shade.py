# -*- coding: utf-8 -*-

import numpy as np
import numba as nb


@nb.njit
def pixel_shading_phong(
    pixel_pos: np.array,
    depth: np.array,
    normal_dir: np.array,
    view_dir: np.array,
    raw_color: np.array,
    lights: np.array
) -> np.array:
    if depth == np.inf:
        return np.ones(shape=(3,))

    ambi_factor = 1.
    spec_factor = .5
    diff_factor = 1.
    
    # [0, 1]
    base_color  = raw_color.astype(np.float64) / 255.
    shade_color = np.zeros_like(base_color)
    
    # for every light
    for light in lights:
        light_color = light[3:]
        light_dir   = light[:3] - pixel_pos
        light_len   = np.linalg.norm(light_dir)
        light_dir   /= light_len

        diff_ratio  = max(np.dot(normal_dir, light_dir), 0.05)
        # halfway
        halfway_dir = light_dir + view_dir
        halfway_dir /= np.linalg.norm(halfway_dir)
        spec_ratio  = max(np.dot(normal_dir, halfway_dir), 0.05) ** 32
        
        # print(diff_ratio, spec_ratio)
        
        ambi_color = ambi_factor * light_color
        diff_color = diff_ratio * diff_factor * light_color
        spec_color = spec_factor * spec_ratio * light_color
        
        shade_color += (ambi_color + diff_color + spec_color) * base_color
    
    return shade_color.clip(0., 1.)

@nb.guvectorize(
    'float64[:,:,:], float64[:,:], float64[:,:,::1], uint8[:,:,:], float64[:,:], uint8[:,:,:]',
    '(n,m,q), (n,m), (n,m,k), (n,m,k), (p,q) -> (n,m,k)',
    nopython=True
)
def shading_phong(ray_buffer: np.array, depth_buffer: np.array, normal_buffer: np.array, raw_colors: np.array, lights: np.array, frame: np.array):
    resol = depth_buffer.shape[0]
    
    for ind in range(resol * resol) :
        row, col = ind % resol, ind // resol
        
        normal_dir = normal_buffer[row, col, :]
        depth = depth_buffer[row, col]
        ray   = ray_buffer[row, col, :]
        pixel_pos = ray[:3] + ray[3:] * depth
        raw_color = raw_colors[row, col, :]
        
        pixel_color = pixel_shading_phong(pixel_pos, depth, normal_dir, -ray[3:], raw_color, lights)
        frame[row, col, :] = (pixel_color * 255).astype(np.uint8)
