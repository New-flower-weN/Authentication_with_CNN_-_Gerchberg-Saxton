import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

i=0
width = height = 1024
dimensions = (width, height)

# test_idx = np.loadtxt("./dataset_64_cifar10/test_indices_25cm.txt").astype(np.int64)

for i in tqdm(range(70000)):

    img = cv2.imread(f"./dataset_32/zero_corr_holo/{100001+i}.png", cv2.IMREAD_GRAYSCALE)

    img1 = np.zeros((1200, 1200))

    # print(img)

    resized_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST) # cv2.INTER_CUBIC
    # img1[(img1.shape[0]//4-resized_img.shape[0]//2):(img1.shape[0]//4+resized_img.shape[0]//2), (img1.shape[1]//4-resized_img.shape[1]//2):(img1.shape[1]//4+resized_img.shape[1]//2)] = resized_img
    # resized_img = np.resize(resized_img, (1, width, height))

    h, w = resized_img.shape
    scale_factor=4

    new_h, new_w = h * scale_factor, w * scale_factor
    # print(f"Размер увеличенного поля: {new_h}x{new_w}")
    offset_x=-w
    offset_y=h
    I_prime = np.zeros((new_h, new_w))  
    center_y, center_x = new_h // 2 + 100, new_w // 2
    start_y = center_y - h // 2 + offset_y
    start_x = center_x - w // 2 #+ offset_x
    start_y = max(0, min(start_y, new_h - h))
    start_x = max(0, min(start_x, new_w - w))
    end_y = start_y + h
    end_x = start_x + w
    # print(f"Размещаю изображение в позиции: y=[{start_y}:{end_y}], x=[{start_x}:{end_x}]")
    I_prime[start_y:end_y, start_x:end_x] = resized_img

    # plt.imshow(I_prime, cmap='gray')
    # plt.show()

    # h, w = img.shape[:2]
    # target_w, target_h = dimensions

    # # Вычисляем коэффициент масштабирования так, чтобы меньшая сторона стала равна целевой
    # # Это гарантирует, что после масштабирования изображение будет покрывать всю целевую область
    # scale = max(target_w / w, target_h / h)
    # new_w = int(w * scale)
    # new_h = int(h * scale)

    # # Масштабируем изображение
    # resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST) # cv2.INTER_CUBIC

    # # Вычисляем координаты для центральной обрезки
    # x_center = new_w // 2
    # y_center = new_h // 2

    # x_start = x_center - target_w // 2
    # y_start = y_center - target_h // 2
    # x_end = x_start + target_w
    # y_end = y_start + target_h

    # # Обрезаем изображение по центру
    # cropped = resized[y_start:y_end, x_start:x_end]

    cv2.imwrite(f"./dataset_32/angled_zero_corr_1200_resized/{100001+i}.png", I_prime) 
