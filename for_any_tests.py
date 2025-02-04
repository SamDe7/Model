# import os 

# for path_dir, dir_list, file_list in os.walk('/Model/MNIST/training'):
#     print(f"Путь к папке {path_dir}")
#     print("Кол-во папок", path_dir.split('/')[-1])
    # print("Кол-во файлов", len(file_list))

from PIL import Image

img = Image.open('/Model/MNIST/training\class_8')
print(img.size)