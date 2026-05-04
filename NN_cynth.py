import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append('./code_for__holo/')
from phys_GS import phys_gs_one

N = 128

# phase1 = cv2.imread('./dataset_64x2_cifar10/almost_const_phase1.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
# phase2 = cv2.imread('./dataset_64x2_cifar10/almost_const_phase2.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) / 255.0
# phases = [phase1, phase2]

model_1 = keras.models.load_model(f"./dataset_classifier/checkpoint/model.h5", compile=False)
model_1.load_weights(f"./dataset_classifier/checkpoint/0100.weights.h5")

for i in tqdm(range(100001, 150001)):
    img = cv2.imread(f'./dataset_classifier/img/{i}.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) 
    img = img / np.max(img)
    input_holo = cv2.imread(f'./dataset_classifier/dataset_1iter/{i}.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) 
    input_holo = input_holo / np.max(input_holo)
    # output_holo = cv2.imread(f'./dataset_64x2_cifar10/holo_all_random/{i}.png', cv2.IMREAD_GRAYSCALE).reshape(1, N, N, 1) 
    # output_holo = output_holo / np.max(output_holo) * 2 * np.pi - np.pi

    input = np.concatenate((img, input_holo), axis=3)

    NN_holo = model_1.predict(input, verbose=0)[0, :, :, 0]

    # NN_hack_holo = phys_gs_one(img[0, :, :, 0], holo, N, N)
    # GS_hack_holo = phys_gs_one(img[0, :, :, 0], output_holo[0, :, :, 0], N, N)

    # print(f'corr phase and NN_hack_holo: {np.corrcoef(input_phase.flatten(), NN_hack_holo.flatten())[0, 1]}\n \
    #       corr phase and holo: {np.corrcoef(input_phase.flatten(), holo.flatten())[0, 1]}\n \
    #       corr phase and GS_hack_holo: {np.corrcoef(input_phase.flatten(), GS_hack_holo.flatten())[0, 1]}\n \
    #       corr phase and output_holo: {np.corrcoef(input_phase.flatten(), output_holo.flatten())[0, 1]}\n ')
    

    cv2.imwrite(f'./dataset_classifier/NN_holo/{i}.png', np.uint8(NN_holo*255))
    # break