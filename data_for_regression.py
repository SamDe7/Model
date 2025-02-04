import os
import numpy as np
import json
from PIL import Image

if not os.path.isdir("datasetset2"):
    os.mkdir("datasetset2")

img = np.random.randint(0, 50, [10, 64, 64], dtype=np.uint8)
square = np.random.randint(100, 200, [10, 15, 15], dtype=np.uint8)

coords = np.empty([10, 2])

data = {}
for i in range(img.shape[0]):
    x = np.random.randint(20, 44)
    y = np.random.randint(20, 44)

    img[i, (y - 7) : (y + 8), (x - 7) : (x + 8)] = square[i]

    coords[i] = [y, x]

    name_img = f"img_{i}.jpg"
    path_img = os.path.join('datasetset2/', name_img)

    image = Image.fromarray(img[i])
    image.save(path_img)

    data[name_img] = [y, x]

    with open('datasetset2/coords.json', 'w') as f:
        json.dump(data, f, indent=5)