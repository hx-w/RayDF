#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import tqdm
from scipy.io import loadmat



class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )

    if len(mesh_filenames) == 0:
        files = list(filter(lambda x: x.endswith('.obj'), os.listdir(shape_dir)))
        if len(files) == 0:
            raise NoMeshFileError()
        else:
            mesh_filenames = [os.path.join(shape_dir, files[0])]
            return mesh_filenames[0]
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_samples_into_ram(filename):
    mat = loadmat(filename)['ray_depth']
    return mat


def unpack_sdf_samples(filename, subsample=None):
    mat = loadmat(filename)['ray_depth']
    if subsample is None:
        return mat

    random_ind = (torch.rand(subsample) * mat.shape[0]).long()
    return mat[random_ind, :]


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    
    random_ind = (torch.rand(subsample) * data.shape[0]).long()
    return data[random_ind, :]
    

class ODFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        tag,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.matfiles = [os.path.join(data_source, tag, f + '.mat') for f in split]

        logging.debug(
            "using "
            + str(len(self.matfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in tqdm.tqdm(self.matfiles, ascii=True):
                # filename = os.path.join(self.data_source, tag, f)
                mat = loadmat(f)['ray_depth']
                self.loaded_data.append(mat)

    def __len__(self):
        return len(self.matfiles)

    def __getitem__(self, idx):
        filename = self.matfiles[idx]
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx
