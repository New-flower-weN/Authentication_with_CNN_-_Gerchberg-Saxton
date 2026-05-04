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
N = 32
N1 = N - 1
ps = 7.8e-6
lmbd = 561.0e-9
# z = np.sqrt(N * N) * ps**2 / lmbd # - формулой описывается оптимальное расстояние или типо того; для моих данных z = 0.2 
z = 0.2
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

# main_phase = cv2.imread('./dataset_64x2_cifar10/Main_phase.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
# phase1 = cv2.imread('./dataset_64x2_cifar10/almost_const_phase1.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
# phase2 = cv2.imread('./dataset_64x2_cifar10/almost_const_phase2.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
# phases = [phase1, phase2]

# algorithm for 1 picture only, img values should be <=1
def phys_gs_one(img, initial_phase, hight, width):
    # creating amplitude from intencivity from initial pic & creating initial random phase distribution over the grid 
    # (doesn't matter if it's from -pi to pi or from 0 to 2pi -> algorithm outputs phase ditribution from -pi to pi)
    img_amp = np.sqrt(img)
    phase = initial_phase
    # phase = np.pi * np.random.uniform(-1, 1, (hight, width)) # np.random.rand(hight, widht)
    # phase = cv2.imread('./dataset_64_cifar10/phase1.png', cv2.IMREAD_GRAYSCALE) / 255.0 * 2 * np.pi - np.pi
    # arr_ssim = np.zeros(iterations) # metric for compering restored & initial pics

    global jj

    for i in range(15):#tqdm(range(iterations)):
        complex_amp = 1 * np.exp(1j * phase)

        freq_img = fftshift(fft2(ifftshift(complex_amp * np.exp(wave_front)))) 

        phase = np.angle(freq_img)
        freq_new = img_amp * np.exp(1j * phase)
        img_plane = ifftshift(ifft2(fftshift(freq_new))) * np.exp(-wave_front) 
        phase = np.angle(img_plane) 

    phase = phase + np.pi

    return np.uint8((phase) * 255 / (2*np.pi))

model_1 = keras.models.load_model(f"./dataset_32/checkpoint/model_epoch_10.h5", compile=False)
# model_1.load_weights(f"./dataset_classifier/checkpoint/0009_512_2ins_1out.weights.h5")

temp_arr = []
temp_corr = []
metric = []

input_arr = []

phase = cv2.imread('./dataset_32/phase_1.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0

np.random.seed(0)
# test_arr = np.loadtxt('./dataset_1024_mnist/test_indices.txt', dtype=np.int64).astype(np.int64) + 10**5

zero_corr = []
zero_nstd = []
corr = []
nstd = []
nstd_true = []
corr_true = []
nstd_holo = []
corr_holo = []

for i in tqdm(range(110001, 110101)):
    idx = i
    # idx = i
    image = cv2.imread(f'./dataset_32/img/{idx}.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
    # img = cv2.imread(f'./dataset_classifier/hack_holo/{idx}.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
    img1 = cv2.imread(f'./dataset_32/zero_corr_holo/{idx}.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
    # img2 = cv2.imread(f'./dataset_classifier/mean_holo_zero_corr.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
    holo = cv2.imread(f'./dataset_32/holo/{idx}.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
    # input_phase = cv2.imread(f'./dataset_classifier/phase.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
    # input_phase = np.random.rand(1, N, N, 1)

    # phase = phases[np.random.randint(0, 2)]
    input1 = np.concatenate((img1, phase), axis=3)
    # input2 = np.concatenate((img, phase), axis=3)
    input_arr.append(input1)

    holo1 = model_1.predict(input1, verbose=0) #+ holo[0, :, :, 0]
    # holo2 = model_1.predict(input2, verbose=0)

    # random_mask = np.random.random(holo1.shape[:2]) < 0.96
    # random_mask = random_mask[:,:,None]  # Добавляем измерение для каналов
    # result = np.where(random_mask, holo1, holo[0, :, :, 0])
    # mask = holo1 * 2 * np.pi > 2 * np.pi
    # holo1[mask] = holo1[mask] - 2 * np.pi
    # holo = phys_gs_one(img[0, :, :, 0], input_phase[0, :, :, 0], N, N) / 255.0

    # print(np.mean(np.abs(holo1 - holo)**2))

    # plt.imshow(input[0, :, :, 0], cmap='gray')
    # plt.show()

    # print(f"corr coeff: {np.corrcoef(holo.flatten(), phase.flatten())}")

    res_nn = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(holo1[0, :, :, 0] * 2*np.pi - np.pi)) * np.exp(wave_front)))))
    res_nn = (res_nn / np.max(res_nn))**2
    plt.imshow(res_nn, cmap='gray')
    plt.show()

    res_true = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(holo[0, :, :, 0] * 2*np.pi - np.pi)) * np.exp(wave_front)))))
    res_true = (res_true / np.max(res_true))**2
    plt.imshow(res_true, cmap='gray')
    plt.show()

    res_zero = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(img1[0, :, :, 0] * 2*np.pi - np.pi)) * np.exp(wave_front)))))
    res_zero = (res_zero / np.max(res_zero))**2
    plt.imshow(res_zero, cmap='gray')
    plt.show()

    print(f"NSTD: {(temp1 := NRMSE(res_zero, image[0, :, :, 0]))}; Corr_coef: {(temp2 := np.corrcoef(res_zero.flatten(), image.flatten())[0, 1])}")
    print(f"NSTD: {(temp3 := NRMSE(res_true, image[0, :, :, 0]))}; Corr_coef: {(temp4 := np.corrcoef(res_true.flatten(), image.flatten())[0, 1])}")
    print(f"NSTD: {(temp5 := NRMSE(res_nn, image[0, :, :, 0]))}; Corr_coef: {(temp6 := np.corrcoef(res_nn.flatten(), image.flatten())[0, 1])}")
    print(f"NSTD: {(temp7 := NRMSE(holo[0, :, :, 0], img1[0, :, :, 0]))}; Corr_coef: {(temp8 := np.corrcoef(holo[0, :, :, 0].flatten(), img1[0, :, :, 0].flatten())[0, 1])}")

    zero_nstd.append(temp1)
    zero_corr.append(temp2)
    nstd_true.append(temp3)
    corr_true.append(temp4)
    nstd.append(temp5)
    corr.append(temp6)
    nstd_holo.append(temp7)
    corr_holo.append(temp8)

    # res2 = np.abs(ifftshift(ifft2(fftshift(np.sqrt(img)[0, :, :, 0] * np.exp(1j*(holo1 * 2*np.pi - np.pi)))))) * np.exp(-wave_front)
    # holo1 = np.angle(res2)
    # res2 = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(holo2[0, :, :, 0]* 2*np.pi - np.pi)) * np.exp(wave_front)))))
    # res2 = (res2 / np.max(res2))**2

    # metric.append(np.std(res2)/np.mean(res2))

    # temp_arr.append(np.corrcoef(res.flatten(), image.flatten())[0, 1])
    # temp_corr.append(np.corrcoef(res2.flatten(), image.flatten())[0, 1])

    # print('true')
    # plt.imshow(image[0, :, :, 0], cmap='gray')
    # plt.show()
    # print('nn')
    # plt.imshow(res, cmap='gray')
    # plt.show()

    # print(f"min: {np.min(holo1)}, max: {np.max(holo1)}")
    # cv2.imwrite(f'./dataset_1024_mnist/res_NNNN{idx}.png', np.uint8(holo1[0, :, :, 0] * 255))

print('zero nstd:', np.mean(zero_nstd), np.std(zero_nstd))
print('zero corr:', np.mean(zero_corr), np.std(zero_corr))
print('nstd true:', np.mean(nstd_true), np.std(nstd_true))
print('corr true:', np.mean(corr_true), np.std(corr_true))
print('nstd nn:', np.mean(nstd), np.std(nstd))
print('corr nn:', np.mean(corr), np.std(corr))
print('nstd holo:', np.mean(nstd_holo), np.std(nstd_holo))
print('corr holo:', np.mean(corr_holo), np.std(corr_holo))

# print(np.mean(temp_arr), np.std(temp_arr))
# print(np.mean(temp_corr), np.std(temp_corr))
# print(np.mean(metric), np.std(metric))

# plt.hist(temp_corr, bins=30, density=True, color='blue', edgecolor='black')
# plt.show()
    # cv2.imwrite(f'./dataset_32/NN_holos_for_hack/{i}.png', holo * 255)

# # Посмотреть все веса модели
# for layer in model_1.layers:
#     if layer.weights:  # если у слоя есть веса
#         print(f"Слой: {layer.name}")
#         for i, weight in enumerate(layer.weights):
#             print(f"  Веса {i}: {weight.name}")
#             print(f"  Форма: {weight.shape}")
#             print(f"  Значения: {weight.numpy()}")
#             print("-" * 50)
