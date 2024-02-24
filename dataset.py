# -*- coding: utf-8 -*-
'''Dataset for RayDF-Net.
'''

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class SimSDFDataset(Dataset):
    def __init__(self, mat_path, batch_max):
        super().__init__()

        samples = loadmat(mat_path)['sdf']
        
        self.coords = samples[:, :3]
        self.depth = samples[:, -1:]
        self.batch_max = batch_max

    def __len__(self):
        return self.coords.shape[0] // self.batch_max

    def __getitem__(self, idx):
        sample_size = self.coords.shape[0]
        # Random coords
        rand_idcs = np.random.choice(sample_size, size=self.batch_max)

        return {
            'coords': torch.from_numpy(self.coords[rand_idcs, :]).float(),
        }, {
            'sdf': torch.from_numpy(self.depth[rand_idcs, :]).float(),
        }

class SimRayDepthDataset(Dataset):
    def __init__(self, mat_path, batch_max):
        super().__init__()

        samples = loadmat(mat_path)['ray_depth']
        
        samples = self._enhance_dataset(samples)
        
        self.coords = samples[:, :3]
        self.dirs = samples[:, 3:-1]
        self.depth = samples[:, -1:]
        self.batch_max = batch_max

    def _enhance_dataset(self, samples: np.array) -> np.array:
        thred = 0.2
        sub = samples[samples[:, -1] < 2.]
        addons = []
        iter_gt = 0
        
        sub_gt = sub[sub[:, -1] > thred]
        while sub_gt.shape[0] > 0:
            sub_gt[:, :3] = sub_gt[:, :3] + thred * sub_gt[:, 3:-1]
            sub_gt[:, -1:] = sub_gt[:, -1:] - thred
        
            addons.append(sub_gt)
            sub_gt = sub_gt[sub_gt[:, -1] > thred]
            iter_gt += 1
        
        addons = np.concatenate(addons, axis=0)
        
        print(f'>> enhance with iter {iter_gt}, add new samples {addons.shape[0]}')
        
        return np.concatenate([addons, samples], axis=0)


    def __len__(self):
        return self.coords.shape[0] // self.batch_max

    def __getitem__(self, idx):
        sample_size = self.coords.shape[0]
        # Random coords
        rand_idcs = np.random.choice(sample_size, size=self.batch_max)

        return {
            'coords': torch.from_numpy(self.coords[rand_idcs, :]).float(),
            'dirs': torch.from_numpy(self.dirs[rand_idcs, :]).float(),
        }, {
            'depth': torch.from_numpy(self.depth[rand_idcs, :]).float(),
        }

class RayDepthDataset(Dataset):
    def __init__(self, mat_path, batch_max, instance_idx=None):
        super().__init__()

        self.instance_idx = instance_idx

        samples = loadmat(mat_path)['ray_depth']
        
        samples = self._enhance_dataset(samples)
        
        self.coords = samples[:, :3]
        self.dirs = samples[:, 3:-1]
        self.depth = samples[:, -1:]
        self.batch_max = batch_max

    def _enhance_dataset(self, samples: np.array) -> np.array:
        thred = 0.2
        sub = samples[samples[:, -1] < 2.]
        addons = []
        iter_gt = 0
        
        sub_gt = sub[sub[:, -1] > thred]
        while sub_gt.shape[0] > 0:
            sub_gt[:, :3] = sub_gt[:, :3] + thred * sub_gt[:, 3:-1]
            sub_gt[:, -1:] = sub_gt[:, -1:] - thred
        
            addons.append(sub_gt)
            sub_gt = sub_gt[sub_gt[:, -1] > thred]
            iter_gt += 1
        
        addons = np.concatenate(addons, axis=0)
        
        print(f'>> enhance with iter {iter_gt}, add new samples {addons.shape[0]}')
        
        return np.concatenate([addons, samples], axis=0)


    def __len__(self):
        return self.coords.shape[0] // self.batch_max

    def __getitem__(self, idx):
        sample_size = self.coords.shape[0]
        # Random coords
        rand_idcs = np.random.choice(sample_size, size=self.batch_max)

        return {
            'coords': torch.from_numpy(self.coords[rand_idcs, :]).float(),
            'dirs': torch.from_numpy(self.dirs[rand_idcs, :]).float(),
            'depth': torch.from_numpy(self.depth[rand_idcs, :]).float(),
            'instance_idx':torch.Tensor([self.instance_idx]).squeeze().long()
        }


class RayDepthMulti(Dataset):
    def __init__(self, root_dir, batch_max, **kwargs):
        #This class adapted from SIREN https://vsitzmann.github.io/siren/

        super().__init__()
        self.root_dir = root_dir
        
        if isinstance(root_dir, list):
            self.instance_dirs = root_dir
        else:
            self.instance_dirs = []
            for file in sorted(os.listdir(root_dir)):
                if file.endswith('mat'):
                    self.instance_dirs.append(os.path.join(root_dir, file))

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        self.all_instances = [
            RayDepthDataset(
                instance_idx=idx,
                mat_path=dir,
                batch_max=batch_max
            ) for idx, dir in enumerate(self.instance_dirs)
        ]

        self.num_instances = len(self.all_instances)
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        ground_truth = [
            { 'depth': obj['depth'] } for obj in observations
        ]

        return observations, ground_truth
