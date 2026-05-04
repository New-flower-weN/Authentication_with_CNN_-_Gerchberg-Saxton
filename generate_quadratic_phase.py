import numpy as np
import cv2
import matplotlib.pyplot as plt

N = 64
a = b = np.pi / N

quad_phase = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        quad_phase[i, j] = a * (i-N/2)**2 + b * (j-N/2)**2

plt.imshow(quad_phase, cmap='gray')
plt.show()

quad_phase = quad_phase / np.max(quad_phase)
quad_phase = np.uint8(quad_phase * 255)
cv2.imwrite('./dataset_64_cifar10/quad_phase.png', quad_phase)
