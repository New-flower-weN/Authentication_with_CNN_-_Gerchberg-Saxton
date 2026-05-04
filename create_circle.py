import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_simple_ring():
    """Простая функция для создания кольца"""
    size = 512
    thickness = 50
    
    # Создаем изображение
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Параметры кольца
    center = (size // 2, size // 2)
    outer_radius = size // 2 - 10
    inner_radius = outer_radius - thickness
    
    # Рисуем кольцо
    cv2.circle(img, center, outer_radius, (255, 255, 255), -1)
    cv2.circle(img, center, inner_radius, (0, 0, 0), -1)
    
    return img

def create_and_show_ring():
    """Создает и показывает кольцо"""
    # Создаем изображение
    img = create_simple_ring()
    
    # Сохраняем
    cv2.imwrite('ring.png', img)
    print("Изображение сохранено как 'ring.png'")
    
    # Показываем через matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Кольцо 512x512, толщина 50px')
    plt.axis('off')
    plt.show()
    
    return img

# Запуск
if __name__ == "__main__":
    img = create_and_show_ring()
    img = np.uint8(((img - np.min(img)) / np.max(img)) * 255)
    # cv2.imwrite("Ring.png", img)