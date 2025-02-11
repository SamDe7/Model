import torch
from torchvision import * 

import torch.nn as nn

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split 
from torchvision.transforms import v2

import os
import json
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

# class MnistDataset(Dataset):
#     def __init__(self, path, transform=None):
#         self.path = path
#         self.transform = transform

#         self.len_dataset = 0
#         self.data_list = []

#         for path_dir, dir_list, file_list in os.walk(path):
#             if path_dir == path:
#                 self.classes = dir_list
#                 self.class_to_idx = {
#                     cls_name : i for i, cls_name in enumerate(self.classes)
#                     }
#                 continue
#             cls = path_dir.split('/')[-1]

            
#             for name_file in file_list:
#                 file_path = os.path.join(path_dir, name_file)
#                 self.data_list.append((file_path, self.class_to_idx[cls]))
        

#             self.len_dataset += len(file_list)

#     def __len__(self):
#         return self.len_dataset

#     def __getitem__(self, index):
#         file_path, target = self.data_list[index]
#         sample = np.array(Image.open(file_path))

#         if self.transform is not None:
#             sample = self.transform(sample)

#         return sample, target

# transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, ), std=(0.5, ))])
transform = v2.Compose([ #функция, позволяющая объединять несколько преобразований в один объект
    v2.Grayscale(num_output_channels=1), #преобразуем цветные изображения в наборе данных к единому цветовому каналу в градациях серого
    v2.ToImage(), #преобразует входные данные в тензор изображения
    v2.ToDtype(torch.float32, scale=True), #преобразует тензор в тип данных float32, а также масштабирует значения пикселей в диапозон значений [0, 1]
    v2.Normalize(mean=(0.5, ), std=(0.5,)) #нормализация тензора изображения по каналам
    ])

train_data = ImageFolder(root='/Model/MNIST/training', transform=transform)
test_data = ImageFolder(root='/Model/MNIST/testing', transform=transform)

# img, cls = test_data[5]
# print(type(img))
# print(img.shape)
# print(img.dtype)
# print(img.min(), img.max())
# print(cls)

# for cls, one_hot in train_data.class_to_idx.items():
#     one_hot_pos = [(i == one_hot)*1 for i in range(10)]
#     print(f"\033[32m{cls}\033[31m => \033[34m{one_hot_pos}\033[0m")

# img, one_hot = train_data[2564]

# cls = train_data.classes[one_hot]
# print(f"Class {cls}")
# plt.imshow(img, cmap='gray')
# plt.show()

train_data, val_data = random_split(train_data, [0.8, 0.2]) #разбитие набора данных на тренировочный и валидационный 

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

image, cls = next(iter(train_loader))

# print(type(image))
# print(image.shape)
# print(image.dtype)
# print('----------')
# print(type(cls))
# print(cls.shape)
# print(cls.dtype)

# for i, (samples, target) in enumerate(train_loader):
#     if i < 3:
#         print(f"Номер батча - {i + 1}")
#         print(f"Размер samples - {samples.shape}")
#         print(f"Размер target - {target.shape}")

# print('\n ................ \n')
# print(f"Номер батча - {i + 1}")
# print(f"Размер samples - {samples.shape}")
# print(f"Размер target - {target.shape}")

# class MyNN(nn.Module): # определяем класс MyNN, который наследуется от класса nn.Module
#     def __init__(self, input, output, hidden_size=2048): # инициализируем класс, принимая параметры: input, output, hidden_size 
#         super().__init__() # вызываем конструктор родительского класса nn.Module
#         layers = [] # создаём пустой список слоёв
#         for _ in range(10): # создаём 10 слоёв, состоящих из линейных преобразований, и функций активации ReLU
#             layers.append(nn.Linear(input, hidden_size)) # добавляем в наш список линейный слой с заданным входом и скрытым размером
#             layers.append(nn.ReLU()) # добавляем слой активации ReLU
#             input = hidden_size # перезаписываем размер входного слоя для следующего
#             hidden_size = int(hidden_size / 2) # уменьшаем скрытый размер вдвое для следующего слоя
#         layers.append(nn.Linear(input, output)) # добаляем последний линейный слой для преобразования в выходной размер
#         self.layers = nn.ModuleList(layers) # сохраняем все слои в виде nn.ModuleList для дальнейшего использования

#     def forward(self, x): # определяем метод прямого прохода с атрибутом x - количество входных данных
#         outputs = [] # создаём список для хранения выходных данных промежуточных слоёв
#         for i, layer in enumerate(self.layers): # проходим по всем слоям и применяем их к входным данным
#             x = layer(x) # применяем текущий слой к входным данным
#             if i != 0 and i % 2 == 0 and i % 4 != 0: # условие для добавления выходных данных промежуточных слоёв в список. Оно нужно, чтобы не добавлять выходные данные со слоёв активации ReLU
#                 outputs.append(x) # добавление выходных данных текущего слоя в список
#             outputs.append(x) # добавление выходных данных последнего слоя отдельно, так как его выход в это условие не попадёт
#         return outputs # возвращение списка выходных данных всех слоёв

# class MyNN(nn.Module):
#     def __init__(self, input, output, hidden_size=2048):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for i in range(10):
#             self.layers.add_module(f"self.layer_{i}", nn.Linear(input, hidden_size))
#             self.layers.add_module(f"act_{i}", nn.ReLU())
#             input = hidden_size
#             hidden_size /= 2
#         self.layers.add_module(f"layer_out", nn.Linear(input, output))

#     def forward(self, x):
#         outputs = []
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i != 0 and i % 2 == 0 and i % 4 != 0:
#                 outputs.append(x)
#             outputs.append(x)
#         return outputs

"""
При такой реализации каждый слой будет иметь своё уникальное название, только добавляться они уже будут не в наш список, а в nn.ModuleList.
"""

# class MyNN(nn.Module):
#     def __init__(self, input, output, hidden_size=2048, choice='rl'):
#         super().__init__()
#         self.activations = nn.ModuleDict({'irl' : nn.LeakyReLU, 'rl' : nn.ReLU})

#         self.layers = nn.ModuleList()
#         for i in range(10):
#             self.layers.add_module(f"self.layer_{i}", nn.Linear(input, hidden_size))
#             self.layers.add_module(f"act_{i}", self.activations[choice])
#             input = hidden_size
#             hidden_size /= 2
#         self.layers.add_module(f"layer_out", nn.Linear(input, output))

#     def forward(self, x):
#         outputs = []
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i != 0 and i % 2 == 0 and i % 4 != 0:
#                 outputs.append(x)
#             outputs.append(x)
#         return outputs

"""
В этом коде был добавлен модуль nn.ModuleDict, в который мы передали 2 вида функций активаци, 
а в классе указали атрибут choice по умолчанию rl. Теперь мы можем удобно менять функции активации на каждом слое, 
обращаясь по ключу.
"""

# model = nn.Sequential(
#     nn.Linear(784, 128),
#     nn.ReLU(),
#     nn.Linear(128, 10)
# )

# input = torch.rand([10, 784], dtype=torch.float32)

# model = nn.Sequential()
# model.add_module('layer_1', nn.Linear(784, 128))
# model.add_module('act_1', nn.ReLU())
# model.add_module('layer_2', nn.Linear(128, 10))

# # input = torch.rand([16, 784])
# # out = model(input)
# # print(out.shape)

# print(model.state_dict())

# class MyNN(nn.Module):
#     def __init__(self, input, output):
#         super().__init__()
#         self.layer_1 = nn.Linear(input, 128)
#         self.act = nn.ReLU()
#         self.layer_2 = nn.Linear(128, output)

#     def forward(self, x, y):
#         x = self.layer_1(x)
#         y = self.act(x + y)
#         out = self.layer_2(y)
#         return out, y

# mod = MyNN(784, 10)

# input = torch.rand([16, 784], dtype=torch.float32)
# input_2 = torch.rand([16, 128])
# out = mod(input, input_2)
# print(len(out))

# class MyNN(nn.Module):
#     def __init__(self, input, output, hidden_size=2048, choice='relu'):
#         super().__init__()
#         self.activate = nn.ModuleDict({
#             'irelu' : nn.LeakyReLU(),
#             'relu' : nn.ReLU()
#         })
#         self.model = nn.ModuleList()
#         for i in range(10):
#             self.model.add_module(f"layer_{i}", nn.Linear(input, hidden_size))
#             self.model.add_module(f"Relu_{i}", self.activate[choice])
#             input = hidden_size
#             hidden_size = int(hidden_size / 2)
#         self.model.add_module(f"output_{i}", nn.Linear(input, output))

#     def forward(self, x):
#         outputs = []
#         for i, layer in enumerate(self.model):
#             x = layer(x)
#             if i != 0 and i % 2 == 0 and i % 4 != 0:
#                 outputs.append(x)
#         outputs.append(x)
#         return outputs

# model = MyNN(784, 2, choice='irelu')
# print(model)
# print(out[0].shape, '\n\n', out[1].shape, '\n\n',out[2].shape, '\n\n', out[3].shape, '\n\n', out[5].shape)

class MyNN(nn.Module):
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

model_classification = MyNN(784, 10)

loss_classication = nn.CrossEntropyLoss()
opt_classification = torch.optim.Adam(model_classification.parameters(), lr=0.001)

model = torch.rand([16, 784], dtype=torch.float32)
out = model_classification(model)
print(out.shape)