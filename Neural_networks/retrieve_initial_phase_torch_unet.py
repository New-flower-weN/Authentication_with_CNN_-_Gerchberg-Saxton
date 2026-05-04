import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from tqdm import tqdm

# Физические параметры (должны быть определены глобально или передаваться)
N = 64
N1 = N - 1
ps = 8.0e-6
lmbd = 633.0e-9
z = np.sqrt(N * N) * ps**2 / lmbd
z2 = 1.5 * z
du = lmbd * z / (N * ps)
du2 = lmbd * z2 / (N * ps)

# Создание сетки для преобразования Фурье
X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
u = ps * X
v = ps * Y
x = du * X
y = du * Y
x2 = du2 * X
y2 = du2 * Y

wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)
wave_front2 = (1j * np.pi / (lmbd * z2)) * (u * u + v * v)

def phys_gs_one_torch_tensor(img_tensor, initial_phase_tensor):
    """
    Физическая модель на чистом PyTorch с поддержкой autograd
    """

    # Создаем волновой фронт как тензор
    device = img_tensor.device
    wave_front_tensor = torch.tensor(wave_front, dtype=torch.complex64).to(device)
    
    # Амплитуда из интенсивности
    img_amp = torch.sqrt(img_tensor)
    phase = initial_phase_tensor
    
    # Итерации физического алгоритма
    for _ in range(1):
        complex_amp = torch.exp(1j * phase) * torch.exp(wave_front_tensor)
        freq_img = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(complex_amp, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))
        phase = torch.angle(freq_img)
        freq_new = img_amp * torch.exp(1j * phase)
        img_plane = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(freq_new, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1)) * torch.exp(-wave_front_tensor)
        phase = torch.angle(img_plane)
    
    phase = phase + torch.pi
    result = (phase) * 255 / (2 * torch.pi) / 255
    return result

# Down sampling block
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        c = F.leaky_relu(self.bn1(self.conv1(x)))
        c = F.leaky_relu(self.bn2(self.conv2(c)))
        p = self.pool(c)
        return c, p

# Up sampling block
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.bn_up = nn.BatchNorm2d(out_channels)
        
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip):
        x = F.leaky_relu(self.bn_up(self.upconv(x)))
        x = torch.cat([x, skip], dim=1)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        return x

# Bottleneck block
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, 512, kernel_size, stride, padding)
        self.bn3 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return x

# U-Net architecture с двумя отдельными путями для выходов
class UNetDualOutput(nn.Module):
    def __init__(self, in_channels=1, image_size=64):
        super(UNetDualOutput, self).__init__()
        self.image_size = image_size
        
        # Down path (общий энкодер)
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        
        # Bottleneck (общий)
        self.bottleneck = Bottleneck(256, 512)
        
        # Up path для первого выхода
        self.up1_output1 = UpBlock(512, 256)
        self.up2_output1 = UpBlock(256, 128)
        self.up3_output1 = UpBlock(128, 64)
        
        # Up path для второго выхода
        self.up1_output2 = UpBlock(512, 256)
        self.up2_output2 = UpBlock(256, 128)
        self.up3_output2 = UpBlock(128, 64)
        
        # Выходные слои
        self.final_conv1 = nn.Conv2d(64, 1, 3, padding=1)  # Первое изображение
        self.final_bn1 = nn.BatchNorm2d(1)
        
        self.final_conv2 = nn.Conv2d(64, 1, 3, padding=1)  # Второе изображение
        self.final_bn2 = nn.BatchNorm2d(1)
        
    def forward(self, x):
        # Encoder (общий для обоих выходов)
        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        
        # Bottleneck (общий)
        bn = self.bottleneck(p3)
        
        # Decoder для первого выхода
        u1_1 = self.up1_output1(bn, c3)
        u2_1 = self.up2_output1(u1_1, c2)
        u3_1 = self.up3_output1(u2_1, c1)
        
        # Decoder для второго выхода
        u1_2 = self.up1_output2(bn, c3)
        u2_2 = self.up2_output2(u1_2, c2)
        u3_2 = self.up3_output2(u2_2, c1)
        
        # Два выхода через разные пути
        output1 = F.relu(self.final_bn1(self.final_conv1(u3_1)))  # Первое изображение
        output2 = F.relu(self.final_bn2(self.final_conv2(u3_2)))  # Второе изображение
        
        return output1, output2

# Функция потерь с физической моделью
def physical_loss(output1, output2, target):
    physical_output = phys_gs_one_torch_tensor(output1, output2)
    mse_loss = F.mse_loss(physical_output, target)
    return mse_loss

# Основная функция обучения
def train_model(train_dataloader, test_dataloader, image_size=64, epochs=10, num_of_folders=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Создание модели с двумя выходами
    model = UNetDualOutput(in_channels=1, image_size=image_size).to(device)
    
    # Оптимизатор
    optimizer = optim.NAdam(model.parameters(), lr=0.001)
    
    # Обучение
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        
        for iter, data in enumerate(train_dataloader):
            data = data.to(device)

            
            optimizer.zero_grad()
            output1, output2 = model(data)
            
            # Используем нашу физическую функцию потерь
            loss = physical_loss(output1, output2, data)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            # if iter % 100 == 0:
            #     print(iter, loss.item())
        
        # Валидация
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data in test_dataloader:
                data = data.to(device)
                output1, output2 = model(data)
                
                val_loss += physical_loss(output1, output2, data).item()
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss/len(train_dataloader):.4f}, '
              f'Val Loss: {val_loss/len(test_dataloader):.4f}, ')
    
    # Сохранение модели
    torch.save(model.state_dict(), f"./dataset_64/{num_of_folders}phase_retrival.weights.pth")
    torch.save(model, f"./dataset_64/{num_of_folders}phase_retrival.pth")
    
    return model

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # или 'RGB' для цветных
        
        if self.transform:
            image = self.transform(image)
            
        return image

if __name__ == "__main__":
    # Преобразования
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # для grayscale
    ])

    # Загрузка всего датасета
    full_dataset = CustomDataset(root_dir='./dataset_64/holo1', transform=transform)

    # Разделение на train и validation
    train_size = 60000
    val_size = 10000
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_model(train_loader, val_loader)