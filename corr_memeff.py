# Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

import tensorflow as tf
from tensorflow import keras

# from holography import func
import cv2

NUM = 6
model_num = int(sys.argv[1])

def NRMSE(x,y):
    mse = np.sum((x - y)**2)
    sum = np.sum(x**2)
    nrmse = np.sqrt(mse/sum)
    return nrmse

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

N = 64
N1 = N - 1
ps = 9.0e-6
lmbd = 532.0e-9
z = (N * ps**2) / lmbd 
# z = 0.04
du = lmbd * z / (N * ps)
k = 2 * np.pi / lmbd

X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
# X, Y = np.meshgrid(np.linspace(-N1 / 2, N1 / 2, 64), np.linspace(-N1 / 2, N1 / 2, 64))
du = lmbd * z / (N * ps)
u = ps * X
v = ps * Y

x = du * X
y = du * Y

wave_front = 1j * np.pi * (u**2 + v**2) / (lmbd * z)

Num_of_pics = 10000

arr_img = np.zeros((Num_of_pics, 64, 64))
arr_holo = np.zeros((Num_of_pics, 64, 64))
j = 0

for i in range(160001, 160001 + Num_of_pics):
    img = cv2.imread(f'./dataset_64/img/{i}.png', cv2.IMREAD_GRAYSCALE)
    holo = cv2.imread(f".\\dataset_64\\holo1\\{i}.png", cv2.IMREAD_GRAYSCALE)

    arr_img[j] = img / 255.0
    arr_holo[j] = holo / 255.0
    j+=1

model = keras.models.load_model(f"./saved_NN/{model_num}phase.h5", compile=False)
# model.compile(optimizer="adam", loss="mean_squared_error", metrics=["root_mean_squared_error"])
model.load_weights(f"./saved_NN/{model_num}phase.weights.h5")

arr_img = np.reshape(arr_img, (Num_of_pics, 64, 64, 1))

result = model.predict(arr_img)[:, :, :, 0]

# print(mas2)
print(result.shape)

corr_list1 = []
corr_list2 = []

# print(np.array(arr_holo).shape)

np.random.seed(0)

for jj in range(100):
    # phase = cv2.imread('./dataset_64/phase_2.png', cv2.IMREAD_GRAYSCALE) / 255.0
    phase = np.random.uniform(0, 1, (64, 64))

    # Рассчет коэффициента корреляции НС голограмм и исх. фазового распределения, но 1 для самого ГС
    for i in range(Num_of_pics):
        corr_list1.append(np.corrcoef(arr_holo[i].flatten(), phase.flatten())[0, 1])
        corr_list2.append(np.corrcoef(result[i].flatten(), phase.flatten())[0, 1])
    
    # np.savetxt(f"./GS/{jj}phase_corr.txt", corr_list1)
    np.savetxt(f"./NN{model_num}/{jj}phase_corr.txt", corr_list2)

    corr_list1.clear()
    corr_list2.clear()
