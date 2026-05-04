# Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# from holography import func
import cv2

def cut(arr_img, start_px, end_px):
    return arr_img[:, start_px:end_px, start_px:end_px]

N = 128
Num_of_pics = 60000
arr_img = np.zeros((Num_of_pics, N, N))
arr_holo = np.zeros((Num_of_pics, N, N))
j = 0

# phase = cv2.imread('./dataset_64x2_cifar10/almost_const_phase1.png', cv2.IMREAD_GRAYSCALE) / 255.0

np.random.seed(0)

for i in range(100001, 160001):
    # idx = np.random.randint(1, 2)
    holo = cv2.imread(f".\\dataset_classifier\\true_holo\\{i}.png", cv2.IMREAD_GRAYSCALE)

    arr_holo[j] = holo / 255.0
    j+=1

# # cutting action
# arr_holo = cut(arr_holo, 16, 48)

mean = np.mean(arr_holo, axis=0)
mean = mean / (np.max(mean) - np.min(mean))
mean = np.uint8(mean * 255)
cv2.imwrite('./dataset_classifier/mean_true_holo.png', mean)

# plt.subplot(1, 2, 1)
# plt.imshow(mean, cmap='gray', vmin=0, vmax=255)
# plt.subplot(1, 2, 2)
# plt.imshow(phase, cmap='gray')
# plt.show()