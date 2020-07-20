'''Data loading utilities.'''
import logging
import tqdm

import numpy as np
import torch
import torch.utils as tutils
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from pathlib import Path

import netCDF4


class DiffusionDataset(Dataset):
    """
    Diffusion dataset.


    Args:
        data_dir (str): Directory with all the data.
        transform (callable): Optional transform to be applied on a sample.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        files_list = os.listdir(data_dir)
        if '.DS_Store' in files_list:
            files_list.remove('.DS_Store')
        self.data_list = sorted(
            files_list, key=lambda x: int(os.path.splitext(x)[0]))
        self.data_len = len(self.data_list)
        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_path = self.data_dir / self.data_list[idx]
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin)

        if self.transform:
            data = self.transform(data)

        return data


class NCEPDataset(Dataset):
    """
    NCEP dataset.


    Args:
        data_dir (str): Directory with all the data.
        transform (callable): Optional transform to be applied on a sample.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        files_list = os.listdir(data_dir)
        if '.DS_Store' in files_list:
            files_list.remove('.DS_Store')
        self.data_list = sorted(
            files_list, key=lambda x: int(os.path.splitext(x)[0]))
        self.data_len = len(self.data_list)
        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_path = self.data_dir / self.data_list[idx]
        data = torch.Tensor(np.load(data_path))

        if self.transform:
            data = self.transform(data)

        # To cater DMM
        data = {
            'observation': data,
            'latent': data
        }

        return data


class CMAPDataset(Dataset):
    """
    CMAP dataset.


    Args:
        data_dir (str): Data location
        transform (callable): Optional transform to be applied on a sample.
    """

    def __init__(self, data, transform=None):
        self.data = data
        self.data_len = len(data)
        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        specified_data = self.data[idx]

        if self.transform:
            specified_data = self.transform(specified_data)

        # To cater DMM
        specified_data = {
            'observation': specified_data,
            'latent': specified_data
        }

        return specified_data


def get_netcdf_data(data_loc, ratio, seq_length, parameter='precip', do_crop=True):
    """
    Get and crop netCDF data from location.

    Args:
        data_loc (Path): Location of the data.
        ratio (float): Ratio of training data.
        seq_length (int): Sequence length of the data.
        parameter (str): Parameter to be extracted as data.
        do_crop (bool): Crop the data or not.

    Return:
        train_data (np.array): Training data.
        valid_data (np.array): Validation data.
    """
    data = netcdf_to_numpy(data_loc, parameter=parameter)
    if do_crop:
        data = crop(data)
    train_data, valid_data = split_data(data, ratio, seq_length)

    return train_data, valid_data


def split_data(data, ratio, seq_length):
    """
    Split data to training and validation minibatches.
    Splitting is done by a moving window mechanism and the data is splitted so that training and validation data don't have any duplicates.
    TODO: Do a version where there are multiple splits


    Args:
        data (np.array): Data to be splitted. Shape: (time, height, width)
        ratio (float): Ratio of splitting.
        seq_length (int): Length of sequence in each data.

    Return:
        train_data (np.array): Splitted training data. Shape: (minibatch, time, height, width)
        valid_data (np.array): Splitted validation data. Shape: (minibatch, time, height, width)
    """
    data_length = data.shape[0] - ((seq_length - 1) * 2)
    train_data_length = int(data_length * ratio)
    split_point = train_data_length + seq_length
    valid_data_length = data_length - train_data_length

    # Make sequences
    train_data_start = 0
    train_data_end = train_data_start + train_data_length
    train_data = [data[i:i + seq_length]
                  for i in range(train_data_start, train_data_end)]

    valid_data_start = train_data_end + seq_length - 1
    valid_data_end = valid_data_start + valid_data_length
    valid_data = [data[i:i + seq_length]
                  for i in range(valid_data_start, valid_data_end)]

    train_data = np.stack(train_data)
    valid_data = np.stack(valid_data)

    return train_data, valid_data


def crop(data):
    """
    Crop 2d data on the center


    Args:
        data (np.array): Data to be cropped. Shape: (time, height, width)

    Return:
        cropped_data (np.array): Cropped data with same size on 1st and 2nd shape.
    """
    height = data.shape[1]
    width = data.shape[2]
    edge_length = int((width - height) / 2)
    start_idx = edge_length
    end_idx = start_idx + height

    return data[:, :, start_idx:end_idx]


def netcdf_to_numpy(data_loc, parameter='precip'):
    """
    Convert NetCDF data to numpy array by extracting the main parameter, which is already in numpy array.


    Args:
        data_loc (Path): Location of the data to be converted.
        parameter (str): Parameter to be extracted.

    Return:
        converted_data (np.array): Converted data.
    """
    datagrp = netCDF4.Dataset(data_loc, 'r', format='NETCDF4')
    assert parameter in [
        variable for variable in datagrp.variables], 'Parameter doesn\'t exist'

    return datagrp.variables[parameter][:, :, :].data


def normalize(data, min_val, max_val):
    """
    Normalize data to -1 and 1.


    Args:
        data (torch.Tensor): Data to be normalized.
        min_val (float): Minimum value.
        max_val (float): Maxmimum value.

    Return:
        data (torch.Tensor): Normalized data.
    """
    return (2.0 * (data - min_val) / (max_val - min_val)) - 1


def denormalize(data, min_val, max_val):
    """
    Denormalize data to -1 and 1.


    Args:
        data (torch.Tensor): Data to be denormalized.
        min_val (float): Minimum value.
        max_val (float): Maxmimum value.

    Return:
        data (torch.Tensor): Denormalized data.
    """
    return ((data * (max_val - min_val)) + max_val + min_val) / 2.0


def main():
    data_loc = Path('./data/cmap/precip.pentad.mean.nc')
    test = netcdf_to_numpy(data_loc)

    train_data, test_data = split_data(crop(test), 0.7, 30)
    print(train_data.shape, test_data.shape)

    # Test
    data = np.array([i for i in range(10)])
    train_data, valid_data = split_data(data, 0.7, 4)


if __name__ == '__main__':
    main()
