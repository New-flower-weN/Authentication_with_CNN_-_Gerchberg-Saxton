import numpy as np
import cv2

img1 = np.zeros((64, 128), dtype=np.uint8)
img2 = np.zeros((64, 128), dtype=np.uint8)

phase1 = cv2.imread('./dataset_64_cifar10/phase1.png', cv2.IMREAD_GRAYSCALE)
phase2 = cv2.imread('./dataset_64_cifar10/phase2.png', cv2.IMREAD_GRAYSCALE)

img1[:, :64] = phase1
img2[:, :64] = phase2

for i in range(100001, 160001):
    temp1= cv2.imread(f'./dataset_64_cifar10/img/{i}.png', cv2.IMREAD_GRAYSCALE)

    img1[:, 64:] = temp1
    img2[:, 64:] = temp1

    cv2.imwrite(f'./dataset_128_64_cifar10/img1/{i}.png', img1)
    cv2.imwrite(f'./dataset_128_64_cifar10/img2/{i}.png', img2)