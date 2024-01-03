# -*- coding: utf-8 -*-

import numpy as np

import utils


if __name__ == '__main__':
    
    A = np.array([1., 2., 3.])
    A /= np.linalg.norm(A)
    
    B = np.array([0., 1., 1.])
    B /= np.linalg.norm(B)

    print('A:', A)
    print('B:', B)

    mat = utils.get_rotation_matrix_from_points(A, B)

    print(A.reshape(1, 3) @ mat.T)
    print(B.reshape(1, 3) @ mat.T)
    
    
    ## mat @ B -> A
