import os
import cv2
import numpy as np

Num_of_pics = 10000

for i in range(150001, 160001):
    os.remove(f'./dataset_classifier/img_1024/{i}.png')

    # idx1 = np.random.randint(100001, 150001 + Num_of_pics)
    # idx2 = np.random.randint(100001, 150001 + Num_of_pics)

    # img1 = cv2.imread(f'./dataset_classifier/zero_corr_holo/{idx1}.png', cv2.IMREAD_GRAYSCALE) / 255.0
    # img2 = cv2.imread(f'./dataset_classifier/zero_corr_holo/{idx2}.png', cv2.IMREAD_GRAYSCALE) / 255.0

    # print(np.corrcoef(img1.flatten(), img2.flatten())[0, 1])