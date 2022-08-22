import numpy as np

import h5py

import torch
import torch.nn as nn


class FoldDataset(torch.utils.data.Dataset):
    dataset_file = None #path to h5py dataset file
    f = None # Dataset file handler (h5py.File in reading mode)

    def __init__(self, set_indexes):
        self.set_indexes = set_indexes

    def __len__(self):
        return len(self.set_indexes)

    def __getitem__(self, index):
        # Data of all sets (train, valid, test) is in one file
        # so we convert the index to match file index
        file_index = self.set_indexes[index]

        # Input features
        x = np.array(self.f['inputs'][file_index], dtype=np.int8)

        # Sample id
        sample = (self.f['samples'][file_index]).astype(np.str_)

        return x, sample


class DAEencoder(nn.Module):
    def __init__(self, n_input_feats, n_hidden_u, bottleneck_size):
        super(DAEencoder, self).__init__()

        # Layers
        self.layers = [nn.Linear(n_input_feats, n_hidden_u[0])]

        for i in range(1,len(n_hidden_u)):
            self.layers.append(nn.Linear(n_hidden_u[i-1], n_hidden_u[i]))

        # Bottleneck layer
        self.layers.append(
                nn.Linear(n_hidden_u[-1], bottleneck_size)
                )

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):
        next_input = x
        for layer in self.layers:
            z = layer(next_input)
            a = torch.relu(z)
            next_input = a

        return a


class DAEdecoder(nn.Module):
    def __init__(self, n_input_feats, n_hidden_u, bottleneck_size):
        super(DAEdecoder, self).__init__()

        # Layers
        self.layers = [nn.Linear(bottleneck_size, n_hidden_u[-1])]

        for i in reversed(range(1,len(n_hidden_u))):
            self.layers.append(nn.Linear(n_hidden_u[i], n_hidden_u[i-1]))

        self.layers.append(nn.Linear(n_hidden_u[0], n_input_feats*3))

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):
        next_input = x
        for layer in self.layers:
            z = layer(next_input)
            a = torch.relu(z)
            next_input = a

        return a

