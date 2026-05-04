import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import torch
from torch import nn
import cv2

# Down sampling block for NN
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        p = self.pool(x)
        return x, p

# First down sampling block for NN
class DownBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        p = self.pool(x)
        return x, p

# Up sampling block for NN
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x

# Bottleneck block for NN
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x

# UNet architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = DownBlock1(1, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 256)
        # self.down5 = DownBlock(64, 128)
        self.bottleneck = Bottleneck(256, 512)
        # self.up1 = UpBlock(256+128, 128)
        self.up2 = UpBlock(512+256, 256)
        self.up3 = UpBlock(256+128, 128)
        self.up4 = UpBlock(128+64, 64)
        self.up5 = UpBlock(64+32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)
        # c5, p5 = self.down5(p4)
        bn = self.bottleneck(p4)
        # u1 = self.up1(bn, c5)
        u2 = self.up2(bn, c4)
        u3 = self.up3(u2, c3)
        u4 = self.up4(u3, c2)
        u5 = self.up5(u4, c1)
        output = self.final(u5)
        return output

    
# Загрузка модели
model = UNet()
model.load_state_dict(torch.load("./UNetW_delete_maybe.pth", weights_only=True))
model.eval()  # Переключение модели в режим оценки

# Параметры
NUM = 5
N = 64
ps = 8e-6 #9.0e-6
lmbd = 633e-9 #532.0e-9
z = (N * ps**2) / lmbd
du = lmbd * z / (N * ps)
# k = 2 * np.pi / lmbd

# Создание сетки
X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
u = ps * X
v = ps * Y
x = du * X
y = du * Y


wave_front = 1j * np.pi * (u**2 + v**2) / (lmbd * z)

# Загрузка изображений
arr_img = np.zeros((15, 64, 64))
arr_holo = np.zeros((15, 64, 64))
j = 0

for i in range(110001, 110015):
    img = cv2.imread(f'C:/Users/Proku/OneDrive/Desktop/kinoform/dataset_64/new_img/{i}.png', cv2.IMREAD_GRAYSCALE)
    holo = cv2.imread(f'C:/Users/Proku/OneDrive/Desktop/kinoform/H_k_64_50_3/K/{i}.png', cv2.IMREAD_GRAYSCALE)
    arr_img[j] = img / 255.0
    arr_holo[j] = holo / 255.0
    j += 1

# Преобразование данных для PyTorch
arr_img = np.reshape(arr_img, (15, 1, 64, 64))  # Добавляем channel dimension
arr_img = torch.tensor(arr_img, dtype=torch.float32)
arr_holo = np.reshape(arr_holo, (15, 1, 64, 64))  # Добавляем channel dimension
arr_holo = torch.tensor(arr_holo, dtype=torch.float32)

# Предсказание с помощью модели
with torch.no_grad():
    result = model(arr_holo)[NUM].squeeze().numpy()  # Убираем batch и channel dimensions

# Визуализация результатов
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# Входное изображение
ax = fig.add_subplot(2, 3, 1)
ax.imshow(arr_img[NUM].squeeze(), cmap="gray")
ax.set_title("Input Image")

# Выходное изображение
ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.abs(result), cmap="gray")
ax.set_title("Generated Image")

cv2.imwrite('holo_output.png', np.uint8(np.abs(result)/np.max(np.abs(result)) * 255))

# Входное изображение (для сравнения)
ax = fig.add_subplot(2, 3, 3)
ax.imshow(arr_holo[NUM].squeeze(), cmap="gray")
ax.set_title("Input Image (Comparison)")

# Восстановленное изображение (FFT)
ax = fig.add_subplot(2, 3, 5)
II = np.abs(fftshift(fft2(ifftshift(np.exp(1j * (result * 2 * np.pi-np.pi)) * np.exp(wave_front)))))**2
ax.imshow(II / np.max(II), cmap='gray')
ax.set_title("Reconstructed Image (FFT)")

# Восстановленное изображение (FFT)
ax = fig.add_subplot(2, 3, 6)
II = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(arr_holo[NUM].squeeze()*2*np.pi-np.pi)) * np.exp(wave_front))))) #**2
ax.imshow(II / np.max(II), cmap='gray')
ax.set_title("Reconstructed Image (FFT)")

plt.show()