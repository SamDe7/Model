from torch import *
from torchvision import * 

from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.transforms import v2
import torch.nn as nn

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DatasetReg(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.list_name_file = os.listdir(path)
        if "coords.json" in self.list_name_file:
            self.list_name_file.remove('coords.json')

        self.len_dataset = len(self.list_name_file)

        with open(os.path.join(self.path, 'coords.json'), 'r') as f:
            self.dict_coords = json.load(f)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        name_file = self.list_name_file[index]
        path_img = os.path.join(self.path, name_file)

        img = Image.open(path_img)
        coord = torch.tensor(self.dict_coords[name_file], dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        return img, coord

transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5], std=[0.5])])

train_data = DatasetReg(path='/Model/Dataset', transform=transform)

# img, cls = train_data[5]
# print(type(img))
# print(img.shape)
# print(img.dtype)
# print(img.min(), img.max())
# print('-----------')
# print(type(cls))
# print(cls.shape)
# print(cls.dtype)
# img, coord = train_data[4352]
# print("Координаты центра", coord)
# plt.scatter(coord[1], coord[0], marker='o', color='red')
# plt.imshow(img, cmap='gray')
# plt.show()

# train_data, val_data, test_data = random_split(train_data, [0.7, 0.1, 0.2])

# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# for i, (sample, target) in enumerate(train_loader):
#     if i < 3:
#         print("Номер батча", i + 1)
#         print("Размер samples", sample.shape)
#         print("Размер target", target.shape)

# print("\n ............. \n")
# print("Номер батча", i + 1)
# print("Размер sample", sample.shape)
# print("размер target", target.shape)

class MyReg(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer_1 = nn.Linear(input, 128)
        self.act = nn.ReLU()
        self.layer_2 = nn.Linear(128, output)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        out = self.layer_2(x)
        return out

model_regression = MyReg(64*64, 2)

loss_regression = nn.MSELoss()
opt_regression = torch.optim.Adam(model_regression.parameters(), lr=0.001)

test = torch.rand([16, 64*64], dtype=torch.float32)
out = model_regression(test)
print(out.shape)