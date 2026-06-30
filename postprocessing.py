import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import cv2
import os

def create_matrix_numpy(rows, cols, zero_fraction=0.5):
    """
    Создает матрицу с заданной долей нулей
    
    Parameters:
    rows, cols - размеры матрицы
    zero_fraction - доля нулей (по умолчанию 0.2 = 20%)
    """
    # Общее количество элементов
    total_elements = rows * cols
    
    # Количество нулей
    num_zeros = int(total_elements * zero_fraction)
    
    # Создаем массив из единиц
    matrix = np.ones((rows, cols), dtype=int)
    
    # Случайным образом выбираем индексы для нулей
    indices = np.random.choice(total_elements, num_zeros, replace=False)
    
    # Преобразуем линейные индексы в двумерные
    row_indices = indices // cols
    col_indices = indices % cols
    
    # Заменяем выбранные элементы на 0
    matrix[row_indices, col_indices] = 0
    
    return matrix

N = 32
ps = 8.0e-6 # исходные ключи исп. 7.8e-6
lmbd = 532.0e-9 # исходные ключи исп. 561.0e-9
# z = np.sqrt(N * N) * ps**2 / lmbd # - формулой описывается оптимальное расстояние или типо того; для моих данных z = 0.2 
z = 0.2
z2 = 1.5 * z
du = lmbd * z / (N * ps)

X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
# X, Y = np.meshgrid(np.linspace(-N1 / 2, N1 / 2, 256), np.linspace(-N1 / 2, N1 / 2, 256))

du = lmbd * z / (N * ps)
u = ps * X
v = ps * Y
x = du * X
y = du * Y

# for shortening creating var that'll be under exp()
wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)

# to_zero = create_matrix_numpy(N, N)
# noise = np.random.rand(1024, 1024)
# cv2.imwrite(f'./dataset_kanji/noise2.png', np.uint8(noise*255))
# noise = cv2.imread('./dataset_kanji/noise2.png', cv2.IMREAD_GRAYSCALE) / 255.0

phase = cv2.imread('./dataset_32/phase_1.png', cv2.IMREAD_GRAYSCALE)
phase = phase - np.mean(phase)
phase = phase / np.std(phase)
phase = phase.flatten()

for i in tqdm(range(100001, 160001)):
    holo = cv2.imread(f'./dataset_32/holo/{i}.png', cv2.IMREAD_GRAYSCALE) / 255.0

    # res = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(holo*2*np.pi-np.pi)) * np.exp(wave_front)))))**2

    # plt.imshow(res)
    # plt.show()

    # holo *= to_zero
    holo = holo - np.mean(holo)
    holo = holo / np.std(holo)
    holo = holo.flatten()

    # print(np.corrcoef(holo, phase)[0, 1])

    holo = holo - np.dot(holo, phase) / np.dot(phase, phase) * phase
    holo = holo.reshape((N,N))
    holo = (holo - np.min(holo)) / (np.max(holo) - np.min(holo))
    # holo = np.sin(holo*np.pi-np.pi/2) * to_zero + np.cos(holo*np.pi) * (1-to_zero)
    # holo = holo + noise * 5
    holo = (holo - np.min(holo)) / (np.max(holo) - np.min(holo))
    holo = np.uint8(holo * 255)

    # res = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(holo/255.0*2*np.pi-np.pi)) * np.exp(wave_front)))))**2
    # res = np.uint8(res / np.max(res) * 255)

    # plt.imshow(res)
    # plt.show()

    # print(len(np.unique(holo)))

    cv2.imwrite(f'./dataset_32/fzero_corr_holo/{i}.png', holo)

# initial = np.zeros((1200, 1200), dtype=np.uint8)

# files = os.listdir("E:\\holos_GS")
# for file in files:
#     img = cv2.imread("E:\\holos_GS\\"+file, cv2.IMREAD_GRAYSCALE)
#     initial[0:img.shape[0], 0:img.shape[1]] = img
#     # print(file.replace("png", "bmp"))
#     cv2.imwrite("E:\\holos_GS\\"+file.replace("png", "bmp"), initial)