# -*- coding: utf-8 -*-

import numpy as np

if __name__ == '__main__':
    S = np.array([
        [2., 0., 0.],
        [0., 1., 0.],
        [0., 0., 2.]
    ])
    
    theta = np.radians(30)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.],
        [0., 0., 1.]
    ])
    
    L = R * S
    
    print(L @ np.array([1, 0, 0]).T)
    
    _U, _S, _V = np.linalg.svd(L)
    
    print(_U @ np.diag(_S) @ _V)
    print(_U)
    print(np.diag(_S))
    # print(_U @ np.diag(_S) @ _V @ np.array([1., 0., 0.]).T)
