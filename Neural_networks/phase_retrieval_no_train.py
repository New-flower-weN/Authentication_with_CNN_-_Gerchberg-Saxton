import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# Определение архитектуры модели (должно быть идентично обучению)
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
    
# Основной код
def main():
    # Пути к файлам
    model_path = "./dataset_64/1phase_retrival.weights.pth"  # или "weights.pth"
    image_path = "./dataset_64/holo1/160002.png"
    # image_path = "./mean.png"
    output_path1 = "output1.jpg"
    output_path2 = "output2.jpg"
    
    try:
        # Определение устройства
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {device}")
        
        # Загрузка модели
        print("Загрузка модели...")
        model = UNetDualOutput(in_channels=1, image_size=64)
        
        # Загрузка весов
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        # Загрузка и предобработка изображения
        print("Загрузка изображения...")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Проверка размера изображения
        if image.shape != (64, 64):
            print(f"Внимание: изображение имеет размер {image.shape}, ожидается 64x64")
            # Если нужно автоматическое изменение размера, раскомментируйте следующую строку:
            # image = cv2.resize(image, (64, 64))
        
        # Нормализация изображения
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5  # нормализация к [-1, 1]
        image_tensor = image_tensor.to(device)
        
        # Прогноз
        print("Обработка нейронной сетью...")
        with torch.no_grad():
            output1, output2 = model(image_tensor)
        
        # Постобработка результатов
        def process_output(output):
            output = output.squeeze().cpu().numpy()
            output = (output * 127.5 + 127.5).astype(np.uint8)  # денормализация к [0, 255]
            return output
        
        result1 = process_output(output1)
        result2 = process_output(output2)
        
        # Сохранение результатов
        cv2.imwrite(output_path1, result1)
        cv2.imwrite(output_path2, result2)
        print(f"Результаты сохранены в {output_path1} и {output_path2}")

        true_phase = cv2.imread("./dataset_64/phase_1.png", cv2.IMREAD_GRAYSCALE) / 255

        # Создаем фигуру с 4 субплoтами
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Отображаем изображения
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(result1, cmap='gray')
        axes[0, 1].set_title('Output 1')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(result2, cmap='gray')
        axes[1, 0].set_title('Output 2')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(true_phase, cmap='gray')
        axes[1, 1].set_title('True Phase')
        axes[1, 1].axis('off')

        # Настраиваем отступы между изображениями
        plt.tight_layout()
        plt.show()
    
        print(np.corrcoef(result1.flatten(), true_phase.flatten())[0, 1])
        print(np.corrcoef(result2.flatten(), true_phase.flatten())[0, 1])

    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()