import kagglehub
import numpy as np
# from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import cv2
import cupy as cp  # Если есть GPU NVIDIA
from cupy.fft import fft2, ifft2, fftshift, ifftshift
import pandas as pd
import deeplake
import glob
import pandas as pd
import re

# ds = deeplake.load("hub://activeloop/kuzushiji-kanji")
# # print(ds)

# images= ds['images']

# dataloader = ds.tensorflow()
# print(dataloader)

# for i, sample in enumerate(images):
#     x = np.array(sample)

#     # cv2.imshow('dafds', x)
#     # cv2.waitKey()

#     #cv2.imwrite(f"./datset_kanji/img_64/{100001+i}.png", x)
#     pathh = 'D:/Person/Prokudin/dataset_kanji/img_64/' + str(1000000 + 1 + i) + '.png'
#     cv2.imwrite(pathh, x)

#     if i > 60000:
#         break

# Download latest version
path = kagglehub.dataset_download("sondosaabed/casia-iris-thousand")

# print("Path to dataset files:", path)

table = pd.read_csv(path + '/iris_thousands.csv')

general_path = path + "/CASIA-Iris-Thousand/"
img_path = table['ImagePath'].apply(lambda string: re.search(r"/casia-iris-thousand/([a-zA-Z/\\\-\d.]*)", string).group(1)).to_list()

# print(img_path[0])

table["Label"].to_csv("./iris_dataset/subject_id.csv", index=False)

for i in tqdm(range(len(img_path))):
    full_path = general_path + img_path[i]
    # print(full_path)
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("fdshfgdasf", img)
    # cv2.waitKey()

    cv2.imwrite(f"./iris_dataset/imgs_small/{100001 + i}.png", img)