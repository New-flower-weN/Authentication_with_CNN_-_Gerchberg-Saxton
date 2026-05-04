# Imports
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

import tensorflow as tf
import keras

# from holography import func
import cv2

NUM = 7

# setting physical parameters
N = 128
N1 = N - 1
ps = 8.0e-6
lmbd = 633.0e-9
z = np.sqrt(N * N) * ps**2 / lmbd # - формулой описывается оптимальное расстояние или типо того; для моих данных z = 0.2 
z2 = 1.5 * z
du = lmbd * z / (N * ps)
du2 = lmbd * z2 / (N * ps)

X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
# X, Y = np.meshgrid(np.linspace(-N1 / 2, N1 / 2, 64), np.linspace(-N1 / 2, N1 / 2, 64))
du = lmbd * z / (N * ps)
u = ps * X
v = ps * Y

x = du * X
y = du * Y

wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)
wave_front2 = (1j * np.pi / (lmbd * z2)) * (u * u + v * v)

arr_img = np.zeros((10, 128, 128, 2))
arr_holo = np.zeros((10, 128, 128))
j = 0

for i in range(120001, 120020, 2):
    # img = cv2.imread(f"./dataset_512/pair_{i}/num_{i}.jpeg", cv2.IMREAD_GRAYSCALE)
    # holo = cv2.imread(f"./dataset_512/pair_{i}/holo_{i}.jpeg", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(f'./dataset_128/img/{i}.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f'./dataset_128/img/{i+1}.png', cv2.IMREAD_GRAYSCALE)
    # holo = cv2.imread(f'./dataset_128/holo/{i}.png', cv2.IMREAD_GRAYSCALE)
    # holo = cv2.imread(f"C:\\Users\\Proku\\OneDrive\\Desktop\\kinoform\\dataset_64\\holo_2_dist\\{i}.png", cv2.IMREAD_GRAYSCALE)

    arr_img[j,:,:,0] = img / 255.0
    arr_img[j,:,:,1] = img2 / 255.0
    # arr_holo[j] = holo / 255.0
    j+=1

j = 0 

for i in range(110001, 110010):
    # img = cv2.imread(f"./dataset_512/pair_{i}/num_{i}.jpeg", cv2.IMREAD_GRAYSCALE)
    # holo = cv2.imread(f"./dataset_512/pair_{i}/holo_{i}.jpeg", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(f'./dataset_128/img/{i}.png', cv2.IMREAD_GRAYSCALE)
    holo = cv2.imread(f'./dataset_128/holo/{i}.png', cv2.IMREAD_GRAYSCALE)
    # holo = cv2.imread(f"C:\\Users\\Proku\\OneDrive\\Desktop\\kinoform\\dataset_64\\holo_2_dist\\{i}.png", cv2.IMREAD_GRAYSCALE)

    # arr_img[j] = img / 255.0
    arr_holo[j] = holo / 255.0
    j+=1

model = keras.models.load_model("./UNet3D.h5", compile=False)
# model.compile(optimizer="adam", loss="mean_squared_error", metrics=["root_mean_squared_error"])
model.load_weights("./UNet3D.weights.h5")

result = model.predict(arr_img)[NUM]

# print(mas2)
print(result.shape)

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# print(result[150])

# ax = fig.add_subplot(2, 3, 1)
# ax.imshow(arr_holo[NUM], cmap="gray")

ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.abs(result), cmap="gray")

ax = fig.add_subplot(2, 3, 1)
ax.imshow(np.abs(arr_img[NUM,:,:,0]), cmap="gray")
ax = fig.add_subplot(2, 3, 4)
ax.imshow(np.abs(arr_img[NUM,:,:,1]), cmap="gray")


ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.abs(arr_holo[NUM]), cmap='gray')

ax = fig.add_subplot(2, 3, 5)
II = np.abs(fftshift(fft2(ifftshift(np.exp(1j * result[:,:,0] * 2 * np.pi) * np.exp(wave_front2)))))**2
ax.imshow(II/np.max(II), cmap='gray')

ax = fig.add_subplot(2, 3, 6)
II = np.abs(fftshift(fft2(ifftshift(np.exp(1j * arr_holo[NUM] * 2 * np.pi) * np.exp(wave_front2)))))**2
ax.imshow(II/np.max(II), cmap='gray')

plt.show()