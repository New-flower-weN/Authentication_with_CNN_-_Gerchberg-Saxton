import cv2
import os
import numpy as np

for i in range(100001, 140001):
    
    os.remove(f'./dataset_64/img/{i}.png')
    
    # img = cv2.imread(f'./dataset_64/new_img/{i}.png', cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite(f'./dataset_64/new_img/{i+1}.png', img)
    # os.remove(f'./dataset_64/img_new/{i}.png')
    # print(np.shape(img))
    # cv2.imwrite(f'./dataset_64/img/{i}.png', img)
    # break