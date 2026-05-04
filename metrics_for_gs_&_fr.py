import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from tqdm import tqdm

def NRMSE(x, y):
    mse = np.sum(x * y)**2
    sum_sq = np.sum(x**2) * np.sum(y**2)
    nrmse = np.sqrt(1 - mse / sum_sq)
    return nrmse

# setting physical parameters
N = 1200
N1 = N - 1
ps = 7.8e-6
lmbd = 561.0e-9
# z = np.sqrt(N * N) * ps**2 / lmbd # - формулой описывается оптимальное расстояние или типо того; для моих данных z = 0.2 
z = 0.4
du = lmbd * z / (N * ps)

# setting the grid for fourier transformation and implementing the phys. parameters like size of pixel for camera/FTML
X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))

du = lmbd * z / (N * ps)
u = ps * X
v = ps * Y
x = du * X
y = du * Y

# for shortening creating var that'll be under exp()
wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)
k = 2 * np.pi / lmbd


# setting physical parameters
NN = 32
NN1 = NN - 1
pss = 7.8e-6
wl = 561.0e-9
# zz = np.sqrt(N * N) * ps**2 / lmbd # - формулой описывается оптимальное расстояние или типо того; для моих данных z = 0.2 
zz = 0.2
duu = wl * zz / (NN * pss)

# setting the grid for fourier transformation and implementing the phys. parameters like size of pixel for camera/FTML
XX, YY = np.meshgrid(np.arange(-NN // 2, NN // 2), np.arange(-NN // 2, NN // 2))

duu = wl * zz / (NN * pss)
uu = pss * XX
vv = pss * YY
xx = duu * XX
yy = duu * YY

# for shortening creating var that'll be under exp()
wave_front2 = (1j * np.pi / (wl * zz)) * (uu * uu + vv * vv)
k2 = 2 * np.pi / wl

corr_holo = []
nstd_holo = []
corr_gs = []
corr_fr = []
nstd_gs = []
nstd_fr = []

model_1 = keras.models.load_model(f"./dataset_32/checkpoint/model_epoch_4.h5", compile=False)
phase = cv2.imread(f"./dataset_32/phase_1.png", cv2.IMREAD_GRAYSCALE) / 255.0
phase = phase.reshape((1, 32, 32, 1))

for i in tqdm(range(100001, 100501)):
    img = cv2.imread(f"./dataset_32/angled_zero_corr_1200_resized/{i}.png", cv2.IMREAD_GRAYSCALE) / 255.0
    img = img[150:450, 150:450]
    cv2.imwrite("original.png", np.uint8(img*255))
    # img = img / np.max(img)
    # img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    # img = img.reshape((1, 32, 32, 1))
    # img = np.concatenate((img, phase), axis=3)

    img = cv2.imread(f"./dataset_32/img/{i}.png", cv2.IMREAD_GRAYSCALE) / 255.0
    holo = cv2.imread(f"./dataset_32/zero_corr_holo/{i}.png", cv2.IMREAD_GRAYSCALE) / 255.0
    # holo = holo.reshape((1, 32, 32, 1))
    # holo = np.concatenate((holo, phase), axis=3)

    gs_holo = cv2.imread(f"./dataset_32/gs_of_angled_1200_01/{i}.png", cv2.IMREAD_GRAYSCALE) / 255.0
    fr_holo = cv2.imread(f"./dataset_32/fresnel_of_angled_1200_01/{i}.png", cv2.IMREAD_GRAYSCALE) / 255.0

    res_gs = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(gs_holo* 2*np.pi - np.pi)) * np.exp(wave_front)))))**2
    res_gs = res_gs[150:450, 150:450]
    res_gs = res_gs - np.min(res_gs)
    res_gs = res_gs / np.max(res_gs)
    # res_gs = cv2.resize(res_gs, (32, 32), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("restored_gs.png", np.uint8(res_gs*255))
    # res_gs = res_gs.reshape((1, 32, 32, 1))
    # res_gs = np.concatenate((res_gs, phase), axis=3)

    c3 = np.exp(1j * k * -z) / (1j * lmbd * -z) * np.exp(1j * np.pi * (x ** 2 + y ** 2) / (lmbd * -z))
    res_fr = np.abs(c3 * ifftshift(ifft2(fftshift(fr_holo * np.exp(1j * np.pi * (u ** 2 + v ** 2) / (lmbd * -z))))))**2
    res_fr = res_fr[:300, :300]
    res_fr = res_fr / np.max(res_fr)
    res_fr = cv2.resize(res_fr, (32, 32), interpolation=cv2.INTER_CUBIC)
    # res_fr = res_fr.reshape((1, 32, 32, 1))
    # res_fr = np.concatenate((res_fr, phase), axis=3)
    # plt.imshow(res_fr)
    # plt.show()

    # img = model_1.predict(img, verbose=0)[0, :, :, 0]
    # holo = model_1.predict(holo, verbose=0)[0, :, :, 0]
    # res_gs = model_1.predict(res_gs, verbose=0)[0, :, :, 0]
    # res_fr = model_1.predict(res_fr, verbose=0)[0, :, :, 0]

    # holo = holo / np.max(holo)
    # res_gs = res_gs / np.max(res_gs)
    # res_fr = res_fr / np.max(res_fr)

    # plt.imshow(holo, cmap='gray')
    # plt.show()

    # plt.imshow(res_gs, cmap='gray')
    # plt.show()
    
    # plt.imshow(res_fr, cmap='gray')
    # plt.show()

    # img = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(img* 2*np.pi - np.pi)) * np.exp(wave_front2)))))**2
    # holo = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(holo* 2*np.pi - np.pi)) * np.exp(wave_front2)))))**2
    # res_gs = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(res_gs* 2*np.pi - np.pi)) * np.exp(wave_front2)))))**2
    # res_fr = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(res_fr* 2*np.pi - np.pi)) * np.exp(wave_front2)))))**2

    print(corr_holo_tmp := np.corrcoef(img.flatten(), holo.flatten()))
    print(corr_gs_tmp := np.corrcoef(holo.flatten(), res_gs.flatten()))
    print(corr_fr_tmp := np.corrcoef(holo.flatten(), res_fr.flatten()))

    print(nstd_holo_tmp := NRMSE(img, holo))
    print(nstd_gs_tmp := NRMSE(img, res_gs))
    print(nstd_fr_tmp := NRMSE(img, res_fr))

    corr_holo.append(corr_holo_tmp)
    corr_gs.append(corr_gs_tmp)
    corr_fr.append(corr_fr_tmp)

    nstd_holo.append(nstd_holo_tmp)
    nstd_gs.append(nstd_gs_tmp)
    nstd_fr.append(nstd_fr_tmp)

    plt.imshow(img, cmap='gray')
    plt.show()

    plt.imshow(holo, cmap='gray')
    plt.colorbar()  # Add color scale for reference
    plt.show()

    plt.imshow(res_gs, cmap='gray')
    plt.colorbar()  # Add color scale for reference
    plt.show()
    
    plt.imshow(holo - res_gs, cmap='gray')
    plt.colorbar()  # Add color scale for reference
    plt.show()

print(f"correlation for holo: {np.mean(corr_holo)} +- {np.std(corr_holo)}")
print(f"correlation for GS: {np.mean(corr_gs)} +- {np.std(corr_gs)}")
print(f"correlation for Fresnel: {np.mean(corr_fr)} +- {np.std(corr_fr)}")

print(f"NSTD for holo: {np.mean(nstd_holo)} +- {np.std(nstd_holo)}")
print(f"NSTD for GS: {np.mean(nstd_gs)} +- {np.std(nstd_gs)}")
print(f"NSTD for Fresnel: {np.mean(nstd_fr)} +- {np.std(nstd_fr)}")