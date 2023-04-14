import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class FlightDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, transform=None):
        """
        Arguments:
            data (numpy array): data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        label = self.data[idx, -1]
        features = self.data[idx, :-1]
        sample = {'features' : features, 'delayed' : label}

        if self.transform:
            sample = self.transform(sample)

        return sample
