import torch
import torchvision

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split 

import os
import json
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

class MnistDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.len_dataset = 0
        self.data_list = []

        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = dir_list
                self.class_to_idx = {
                    cls_name : i for i, cls_name in enumerate(self.classes)
                                    }
                continue
            cls = path_dir.split('/')[-1]

            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_idx[cls]))

            self.len_dataset += len(file_list)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        sample = np.array(Image.open(file_path))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

train_data = MnistDataset('/Model/MNIST/training')
test_data = MnistDataset('/Model/MNIST/raw')

print(train_data.classes)