import torch
from torchvision import *
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder

from torchvision.transforms import v2

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

# plt.axis('off')
# img = np.array(Image.open('/Model/images/beta_test.jpg'))

# transform = transforms.ToTensor()
# img_to_tensor = transform(img)


# transform = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

# img_norm = transform(img_to_tensor)
# print(type(img_norm))
# print(img_norm.shape)
# print(img_norm.dtype)
# print(img_norm.min(), img_norm.max())

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5, ), std=(0.5, 0.5, 0.5))])
# img = transform(Image.open('/Model/images/beta_test.jpg'))

# transform = v2.ToImage()
# img = transform(Image.open('/Model/images/beta_test.jpg'))

# transform = v2.ToDtype(torch.float32, scale=True)
# img2 = transform(img)
# transform = v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# img_v2 = transform(img2)

img = np.array(Image.open('/Model/images/beta_test.jpg'))
transform = transforms.Compose([transforms.ToTensor(), v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
img_v2 = transform(img)

print(type(img_v2))
print(img_v2.shape)
print(img_v2.dtype)
print(img_v2.min(), img_v2.max())