import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras

def NRMSE(x, y):
    mse = np.sum(x * y)**2
    sum_sq = np.sum(x**2) * np.sum(y**2)
    nrmse = np.sqrt(1 - mse / sum_sq)
    return nrmse

def plot(data, mean_data, sigma_data, name):
    plt.hist(data, bins=30, density=False, color='blue', edgecolor='black')

    plt.axvline(mean_data, color='red', linestyle='dashed', linewidth=1.5, label=f'mean: {mean_data}')

    plt.axvline(mean_data + sigma_data, color='green', linestyle='dashed', linewidth=2, label=f'mean+sigma\nsigma: {sigma_data}')
    plt.axvline(mean_data - sigma_data, color='green', linestyle='dashed', linewidth=2, label=f'mean-sigma\nsigma: {sigma_data}')

    plt.axvline(mean_data + 2*sigma_data, color='orange', linestyle='dashed', linewidth=2, label='mean+2sigma')
    plt.axvline(mean_data - 2*sigma_data, color='orange', linestyle='dashed', linewidth=2, label='mean-2sigma')

    plt.axvline(mean_data + 3*sigma_data, color='purple', linestyle='dashed', linewidth=2, label='mean+3sigma')
    plt.axvline(mean_data - 3*sigma_data, color='purple', linestyle='dashed', linewidth=2, label='mean-3sigma')

    plt.xlabel('Value')
    plt.ylabel('Num of examples')
    plt.legend()
    plt.grid(True)
    # plt.savefig(f'{name}.png')
    plt.show()

N = 64
Num_of_pics = 10000

corr_arr_same = []
corr_arr_diff = []

nrmse_arr_same = []
nrmse_arr_diff = []

np.random.seed(0)

for i in range(10000):
    idx1 = np.random.randint(150001, 150001 + Num_of_pics)
    idx2 = np.random.randint(150001, 150001 + Num_of_pics)

    holo1_NN = cv2.imread(f'./dataset_64x2_cifar10/holo_almost_const3/{idx1}.png', cv2.IMREAD_GRAYSCALE) / 255.0
    pair_1 = cv2.imread(f'./dataset_64x2_cifar10/holo_almost_const3/{idx2}.png', cv2.IMREAD_GRAYSCALE) / 255.0
    pair_2 = cv2.imread(f'./dataset_64x2_cifar10/holo_almost_const4/{idx2}.png', cv2.IMREAD_GRAYSCALE) / 255.0

    corr_arr_same.append(np.corrcoef(holo1_NN.flatten(), pair_1.flatten())[0, 1])
    corr_arr_diff.append(np.corrcoef(holo1_NN.flatten(), pair_2.flatten())[0, 1])

    nrmse_arr_same.append(NRMSE(holo1_NN, pair_1))
    nrmse_arr_diff.append(NRMSE(holo1_NN, pair_2))

corr_mean_same = np.mean(corr_arr_same)
corr_mean_diff = np.mean(corr_arr_diff)
corr_sigma_same = np.std(corr_arr_same)
corr_sigma_diff = np.std(corr_arr_diff)

nrmse_mean_same = np.mean(nrmse_arr_same)
nrmse_mean_diff = np.mean(nrmse_arr_diff)
nrmse_sigma_same = np.std(nrmse_arr_same)
nrmse_sigma_diff = np.std(nrmse_arr_diff)

plot(corr_arr_same, corr_mean_same, corr_sigma_same, 'corr_same')
plot(corr_arr_diff, corr_mean_diff, corr_sigma_diff, 'corr_diff')

plot(nrmse_arr_same, nrmse_mean_same, nrmse_sigma_same, 'nrmse_same')
plot(nrmse_arr_diff, nrmse_mean_diff, nrmse_sigma_diff, 'nrmse_diff')