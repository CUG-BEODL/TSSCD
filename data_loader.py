"""
@Author ：hhx
@Description ：DataLoader
"""
from torch.utils import data
import numpy as np
import pandas as pd
import random
import glob
import os


class MaskDataset(data.Dataset):
    def __init__(self, paths, type):
        super(MaskDataset, self).__init__()
        self.image_paths = paths
        self.type = type

    def __getitem__(self, index):
        data, label = self.image_paths[index, :-1], self.image_paths[index, -1]
        if self.type == 'train':
            """if train dataset, then apply data enhancement"""
            if random.random() < 0.5:
                data = np.flip(data, axis=1).copy()
                label = label[::-1].copy()
        else:
            pass
        label[label == 6] = 5


        return data, label

    def __len__(self):
        return self.image_paths.shape[0]


def load_data():
    train_ = np.load('dataset/train.npy').transpose(0, 2, 1)
    test_ = np.load('dataset/test.npy').transpose(0, 2, 1)


    train_ds = MaskDataset(paths=train_, type='train')
    test_ds = MaskDataset(paths=test_, type='test')

    train_dl = data.DataLoader(dataset=train_ds, batch_size=64, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=64)

    return train_dl, test_dl