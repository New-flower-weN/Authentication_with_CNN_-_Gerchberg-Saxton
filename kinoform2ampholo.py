import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import cv2

# setting physical parameters
N = 128
N1 = N - 1
ps = 5.4e-6
lmbd = 639.0e-9
z = np.sqrt(N * N) * ps**2 / lmbd # - формулой описывается оптимальное расстояние или типо того; для моих данных z = 0.2 
du = lmbd * z / (N * ps)

phase = cv2.imread('./dataset_classifier/true_holo/100001.png', cv2.IMREAD_GRAYSCALE) / 255.0 * 2 * np.pi - np.pi

X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))

du = lmbd * z / (N * ps)
u = ps * X
v = ps * Y
x = du * X
y = du * Y

wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)

complex_amp = 1 * np.exp(1j * phase)

