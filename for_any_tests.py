# import os

# for path_dir, dir_list, file_list in os.walk('/Model/MNIST/training'):
#     print(f"The path to {path_dir}")
#     print("The class", path_dir.split('/')[-1])
#     print("The number of blobs", len(file_list))
# 
# import os 

# os.path.join('/Model/MNIST/training/class_9', '4.jpg')
import numpy as np

# arr = np.array([[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]])

# new = arr.view()
# new[0, 1] = 7
# print(arr, '\n\n', new)

# arr = np.array([[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]])

# new = arr.copy()
# new[0, 1] = 5
# print(arr, '\n\n', new)

# arr = np.array([[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]])
# arr2 = np.array([[[1, 2, 4, 5], [2, 3, 6, 7], [3, 4, 8, 9]]])
# arr3 = np.array([[[1, 2, 3, 4, 5], [2, 3, 4, 5, 5], [3, 4, 5, 6, 6]]])
# new = np.expand_dims(arr, axis=2)
# arr = arr[..., np.newaxis]
# arr = np.squeeze(arr, axis=0)
# new = np.delete(arr, 1, axis=2)
# united = np.concatenate(arr, arr2, axis=2)
# united2 = np.vstack(united, arr3, axis=1)
# print(united2.shape, united2, sep='\n\n')

# arr1 = np.empty([27, 8, 3])
# arr2 = np.empty([27, 10, 3])
# arr3 = np.empty([27, 9, 3])
# res = np.hstack([arr1, arr2, arr3])
# print(res.shape)

# arr1 = np.empty([8, 27, 3])
# arr2 = np.empty([10, 27, 3])
# arr3 = np.empty([9, 27, 3])
# res = np.vstack([arr1, arr2, arr3])
# print(res.shape)

# arr1 = np.empty([8, 27, 3])
# arr2 = np.empty([10, 27, 3])
# arr3 = np.empty([9, 27, 3])
# res = np.concatenate((arr1, arr2, arr3), axis=0)
# print(res.shape)

# st = np.array([['x_y', 'y_x'], ['x_y', 'y_x'], ['x_y', 'y_x'], ['x_y', 'y_x']])
# st_spl = np.split(st, 2, axis=1)
# st_spl[1][0] = 1
# print(st)

# data = np.empty([1000, 28, 28])
# arr1 = np.empty([50, 28, 28])
# arr2 = np.empty([28, 28])

# # new_data = np.append(data, arr1)
# # new_data = np.append(data, arr1, axis=0)
# # new_data = np.append(data, arr2[np.newaxis], axis=0)
# new_data = np.delete(data, 500, axis=0)
# print(new_data.shape)

# import time
# data = np.empty([100, 28, 28])
# start = time.time()
# for i in range(100):
#     img = np.random.randint(0, 255, 28*28).reshape([28, 28])
#     data = np.append(data, img[np.newaxis], axis=0)
# end = time.time()
# res = end - start
# print(data.shape, '\n', res)

# import time
# start = time.time()
# data = np.empty([100, 28, 28])
# data.resize([200, 28 ,28])

# for i in range(100):
#     new_data = np.random.randint(0 ,255, 28*28).reshape([28, 28])
#     data[100+i] = new_data
# end = time.time()
# res = end - start
# print(res)

# arr = np.array([[[1, 2, 3], [7, 8 ,9], [4, 5, 6]]])
# print(np.sum(arr, axis=2))

# arr = np.arange(3*4*4*3).reshape([3, 4, 4, 3])
# # arr.shape = [3, 4*4, 3]
# print(np.mean(arr, axis=(0, 1, 2)))

# arr = np.arange(3*4*4*3).reshape([3, 4, 4, 3])
# print(np.std(arr))
# import matplotlib.pyplot as plt
# p = np.random.normal(loc=5, scale=1, size=100)
# plt.hist(p)
# plt.show()

# arr = np.arange(15).reshape(3, 5)
# # print(np.random.choice(arr, 15))
# np.random.shuffle(arr)
# print(arr)

# np.random.seed(42)
# arr = np.random.rand(3)

# print(arr)

# import numpy as np

# arr1 = np.array([[1, 2, 3], [4, 5, 6]])
# arr2 = np.array([[1, 2, 3], [4, 10, 6]])
# arr_emp = []

# for i in arr1:
#     for j in arr2:
#         if np.array_equal(i, j):
#             arr_emp.append(i)
        

# print(arr_emp)

# arr1 = np.array([[1, 2, 3], [4, 5, 6]])
# arr2 = np.array([[1, 3, 3], [4, 10, 6]])

# # print(np.equal(arr1, arr2))
# print(np.less(arr2, arr1))

import numpy as np

# def compare():
#     arr1 = np.array([[1, 2, 3], [4, 5, 6]])
#     arr2 = np.array([[1, 3, 3], [4, 10, 6]])

#     # Находим элементы, которые не равны
#     not_equal = np.not_equal(arr1, arr2)
    
#     # Заменяем элементы в обоих массивах
#     arr1[not_equal] = 100
#     arr2[not_equal] = 100

#     return arr1, arr2

# result1, result2 = compare()
# print(result1)
# print(result2)

# arr = np.random.randint(10, size=[2, 3, 2])
# shape_2 = np.arange(4, 10).reshape(2, 3)
# shape_3 = np.array([6 ,7])

# # compare = arr == shape_2.reshape(2, 3, 1)
# print(arr < shape_3[:, np.newaxis, np.newaxis])

# arr = np.array([1, 2, 3, 4, 5])
# print(np.all(arr < 3))
# print(np.any(arr > 4))

# vector1 = np.array([1, 2, 3, 4, 5])
# vector2 = np.array([2, 3, 4, 5, 6])

# mat1 = np.random.randint(10, size=[2, 5])
# mat2 = np.random.randint(10, size=[2, 2])

# print(np.dot(vector1, vector2))
# print(vector1 @ vector2)        # это всё для скалярного произведения векторов
# print(np.inner(vector1, vector2))

# print(np.outer(vector2, vector1)) # внешнее произведение векторов

# print(np.dot(mat1, mat2))
# print(np.matmul(mat1, mat2))  # это всё умножение матриц между собой
# print(mat1 @ mat2)


# print(mat1 @ vector1)
# print(np.dot(vector2, mat1))

# print(np.linalg.trace(mat2)) # вычисляет след матрицы

mat = np.array([[3, 4, 5], [5, 6, 7], [8, 9, 11]])
# print(np.linalg.matrix_rank(mat))
print(np.linalg.inv(mat)) #нахождение обратной матрицы
# print(np.linalg.det(mat)) # для нахождения опредлителя матрицы