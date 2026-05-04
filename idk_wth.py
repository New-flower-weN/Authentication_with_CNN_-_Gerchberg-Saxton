# Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from tqdm import tqdm
import cv2

def NRMSE(x, y):
    mse = np.sum(x * y)**2
    sum_sq = np.sum(x**2) * np.sum(y**2)
    nrmse = np.sqrt(1 - mse / sum_sq)
    return nrmse

N = 32
pixel_size = 8.0e-6
wave_length = 633.0e-9
z = np.sqrt(N * N) * pixel_size**2 / wave_length
du = wave_length * z / (N * pixel_size)
initial_phase = 2 * np.pi * cv2.imread('./dataset_32/phase_1.png', cv2.IMREAD_GRAYSCALE) / 255.0 - np.pi

X, Y = np.meshgrid(np.arange(-N//2, N//2), np.arange(-N//2, N//2))

du = wave_length * z / (N * pixel_size)
u = pixel_size * X
v = pixel_size * Y
x = du * X
y = du * Y

wave_front = (1j * np.pi / (wave_length * z)) * (u * u + v * v)

class phys_GS_nrmse():
    def __init__(self, arr_img, it):
        self.arr_img = np.sqrt(arr_img)
        self.it = it
        self.nstd = []
        self.num_images = arr_img.shape[0]
        self.image_shape = arr_img.shape[1:]

    def algorithm(self):
        # Initialize phase for each image
        phase = np.tile(initial_phase, (self.num_images, 1, 1))
        
        # print(phase[0] - initial_phase, phase[5] - initial_phase)

        for _ in range(self.it):
            complex_amp = np.exp(1j * phase)
            
            # Apply wavefront correction and perform FFT for each image
            freq_img = np.zeros_like(complex_amp, dtype=complex)
            corrected_wave = complex_amp * np.exp(wave_front)
            freq_img = fftshift(fft2(ifftshift(corrected_wave, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))
            
            phase = np.angle(freq_img)
            
            # Apply measured amplitude and perform IFFT for each image
            freq_new = self.arr_img * np.exp(1j * phase)
            img_plane = np.zeros_like(freq_new, dtype=complex)
            img_plane = ifftshift(ifft2(fftshift(freq_new, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)) * np.exp(-wave_front)
            
            phase = np.angle(img_plane)
        
        phase = phase + np.pi

        # print(np.max(phase), np.max(initial_phase))
        
        # Calculate correlation for each image
        for i in range(self.num_images):
            # nrmse = NRMSE(phase[i] / (2*np.pi), (initial_phase + np.pi)/(2*np.pi))
            nrmse = np.corrcoef((phase[i] / (2*np.pi)).flatten(), ((initial_phase + np.pi)/(2*np.pi)).flatten())[0][1]
            self.nstd.append(nrmse)  
        
        print(f"{self.it}: {np.std(self.nstd)}")

        return np.mean(self.nstd) #, np.abs(freq_img)**2

# Load images
arr = []
for i in range(150001, 151001):
    img = cv2.imread(f'./dataset_32/img/{i}.png', cv2.IMREAD_GRAYSCALE) / 255.0
    arr.append(img)

arr = np.array(arr)
nrmse_arr = []

for i in range(20):
    print(f"Iteration {i}")
    gs = phys_GS_nrmse(arr, i)
    nrmse_arr.append(gs.algorithm())

print(nrmse_arr)
plt.plot(nrmse_arr)
plt.xlabel('Iterations')
plt.ylabel('Average NSTD')
plt.title('GS Algorithm Convergence')
plt.show()

# img = cv2.imread('./dataset_32/img/149999.png', cv2.IMREAD_GRAYSCALE) / 255.0
# print(NRMSE(img, (initial_phase + np.pi)/(2*np.pi)))
# img = img.reshape(1, 32, 32)

# img_holo = cv2.imread('./dataset_32/holo1/149999.png', cv2.IMREAD_GRAYSCALE) / 255.0
# print(NRMSE(img, (initial_phase + np.pi)/(2*np.pi)))
# img_holo = img_holo.reshape(1, 32, 32)

# # Посмотрите на разные изображения
# plt.figure(figsize=(15, 5))

# plt.subplot(131)
# plt.imshow(img_holo[0], cmap='gray')  # Голограмма
# plt.title('Голограмма истинная')

# plt.subplot(132)
# plt.imshow(phys_GS_nrmse(img, 15).algorithm()[1][0], cmap='gray')  # Истинная фаза
# plt.title('Голограмма синтезированнная')

# plt.subplot(133)
# plt.imshow((initial_phase+np.pi)/(2*np.pi), cmap='gray')  # Восстановленная фаза
# plt.title('Начальная фаза')

# plt.show()