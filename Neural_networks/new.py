import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

import torch
import torch.nn as nn
import torch.fft
   
# UNet parameters
epochs = 8
BATCH_SIZE = 1

# Loading img
def load_image(image_file):
    image = Image.open(image_file).convert('L')
    image = transforms.ToTensor()(image)
    image = image / 255.0

    return image

# Loading img & img-label
def load_image_pair(dir_path):
    input_image = load_image(dir_path)
    label_image_path = str(dir_path)[:40] + '/H_k_64_50_3/K' + str(dir_path)[59:]
#    print(label_image_path)
    label_image = load_image(label_image_path)
    # plt.imshow(label_image[0])
    # plt.show()

    return label_image, input_image 

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return load_image_pair(self.file_paths[idx])

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

# Create dataset and dataloader
train_img = list(pathlib.Path('C:/Users/Proku/OneDrive/Desktop/kinoform/dataset_64/new_img/').glob('*'))
train_dataset = CustomDataset(train_img[:10000])
test_dataset = CustomDataset(train_img[10000:11000])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Инициализация модели, функции потерь и оптимизатора
model = UNet()
criterion = nn.L1Loss()
criterion_mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Добавляем scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.75)  # * 0.75

i = 0

# Training the model
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Вычисляем потерю
        loss = criterion_mse(outputs, labels)
        
        # Обратное распространение и обновление весов
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print('Instant loss:', loss.item())
        i += 1

    # Валидация
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            val_loss += criterion_mse(outputs, labels).item()
        val_loss /= len(test_loader)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]}')

    # Обновляем learning rate
    scheduler.step()

# Save the model
torch.save(model.state_dict(), "./UNetW_delete_maybe.pth")