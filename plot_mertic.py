import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras

def NRMSE(x, y):
    mse = np.sum(x * y)**2
    sum_sq = np.sum(x**2) * np.sum(y**2)
    nrmse = np.sqrt(1 - mse / sum_sq)
    return nrmse

def plot(data, mean_data, sigma_data, name):
    plt.hist(data, bins=30, density=False, color='blue', edgecolor='black')

    plt.axvline(mean_data, color='red', linestyle='dashed', linewidth=1.5, label=f'mean: {mean_data}')

    plt.axvline(mean_data + sigma_data, color='green', linestyle='dashed', linewidth=2, label=f'mean+sigma\nsigma: {sigma_data}')
    plt.axvline(mean_data - sigma_data, color='green', linestyle='dashed', linewidth=2, label=f'mean-sigma\nsigma: {sigma_data}')

    plt.axvline(mean_data + 2*sigma_data, color='orange', linestyle='dashed', linewidth=2, label='mean+2sigma')
    plt.axvline(mean_data - 2*sigma_data, color='orange', linestyle='dashed', linewidth=2, label='mean-2sigma')

    plt.axvline(mean_data + 3*sigma_data, color='purple', linestyle='dashed', linewidth=2, label='mean+3sigma')
    plt.axvline(mean_data - 3*sigma_data, color='purple', linestyle='dashed', linewidth=2, label='mean-3sigma')

    plt.xlabel('Value')
    plt.ylabel('Num of examples')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{name}.png')
    plt.show()

def cut(arr_img, start_px, end_px):
    return arr_img[:, start_px:end_px, start_px:end_px]

N = 128
Num_of_pics = 10000

arr_holomean = np.zeros((Num_of_pics, N, N))
arr_holomean_nn = np.zeros((Num_of_pics, N, N))
arr_holo1 = np.zeros((Num_of_pics, N, N, 1))
arr_holo2 = np.zeros((Num_of_pics, N, N, 1))
arr_img = np.zeros((Num_of_pics, N, N, 1))
arr_GShack = np.zeros((Num_of_pics, N, N))
phase_arr = np.zeros((Num_of_pics, N, N, 1))

GShack_corr = []
GShack_nrmse = []
corr_phase_GS = []
corr_phase_NN = []
corr_NN_GS = []
nrmse_phase_GS = []
nrmse_phase_NN = []
nrmse_NN_GS = []
corr_leftholo_GS = []
corr_leftholo_NN = []
corr_leftholo_phase = []
nrmse_leftholo_GS = []
nrmse_leftholo_NN = []
nrmse_leftholo_phase = []
j = 0

phase1 = cv2.imread(f'./dataset_64x2_cifar10/almost_const_phase1.png', cv2.IMREAD_GRAYSCALE)
phase1 = phase1 / np.max(phase1)   
phase2 = cv2.imread(f'./dataset_64x2_cifar10/almost_const_phase2.png', cv2.IMREAD_GRAYSCALE)
phase2 = phase2 / np.max(phase2)   

phases = [phase1, phase2]

model_1 = keras.models.load_model(f"./dataset_classifier/checkpoint/model.h5", compile=False)
model_1.load_weights(f"./dataset_classifier/checkpoint/0100_holo2holo.weights.h5")

for i in range(150001, 150001 + Num_of_pics):
    img = cv2.imread(f'./dataset_classifier/img/{i}.png', cv2.IMREAD_GRAYSCALE).reshape(N, N, 1)
    img = img / np.max(img)
    holo1 = cv2.imread(f".\\dataset_classifier\\dataset_1iter\\{i}.png", cv2.IMREAD_GRAYSCALE).reshape(N, N, 1)
    holo1 = holo1 / np.max(holo1)
    holo2 = cv2.imread(f".\\dataset_classifier\\false_holo\\{i}.png", cv2.IMREAD_GRAYSCALE).reshape(N, N, 1)
    holo2 = holo2 / np.max(holo2)
    GShack = cv2.imread(f".\\dataset_classifier\\GS_hack\\{i}.png", cv2.IMREAD_GRAYSCALE) 
    GShack = GShack / np.max(GShack)
    
    # phase = cv2.imread(f".\\dataset_classifier\\phases_all_random\\{i}.png", cv2.IMREAD_GRAYSCALE).reshape(N, N, 1)
    phase = cv2.imread(f".\\dataset_classifier\\rand_phase.png", cv2.IMREAD_GRAYSCALE).reshape(N, N, 1)
    phase_arr[j] =  phase / np.max(phase)
    arr_img[j] = img
    arr_holo1[j] = holo1 
    arr_holo2[j] = holo2 
    arr_GShack[j] = GShack
    j+=1

input_tensor = tf.concat([arr_img, arr_holo1], axis=-1)

# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4) 

# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(phase_arr[0, :, :, 0], cmap="gray")

# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(input_tensor[0, :, :, 1], cmap="gray")

# plt.show()

# holo predict
arr_holo1_nn = model_1.predict(input_tensor)

# # cutting action for edge of imgs 
# arr_holo1 = cut(arr_holo1, 0, 32)
# arr_holo2 = cut(arr_holo2, 0, 32)
# arr_holo1_nn = cut(arr_holo1_nn, 0, 32)
# phase = phase[0:32, 0:32]
# arr_GShack = cut(arr_GShack, 0, 32)

# # cutting action for center of imgs
# arr_holo1 = cut(arr_holo1, 16, 48)
# arr_holo2 = cut(arr_holo2, 16, 48)
# arr_holo1_nn = cut(arr_holo1_nn, 16, 48)
# phase = phase[16:48, 16:48]
# arr_GShack = cut(arr_GShack, 16, 48)

# calculate NRMSE & correlation
for i in range(Num_of_pics):
    #right holo
    corr_phase_GS.append(np.corrcoef(phase_arr[i, :, :, 0].flatten(), arr_holo1[i, :, :, 0].flatten())[0, 1])
    corr_phase_NN.append(np.corrcoef(phase_arr[i, :, :, 0].flatten(), arr_holo1_nn[i, :, :, 0].flatten())[0, 1])
    corr_NN_GS.append(np.corrcoef(arr_holo1_nn[i, :, :, 0].flatten(), arr_holo1[i, :, :, 0].flatten())[0, 1])

    nrmse_phase_GS.append(NRMSE(phase_arr[i, :, :, 0], arr_holo1[i, :, :, 0]))
    nrmse_phase_NN .append(NRMSE(phase_arr[i, :, :, 0], arr_holo1_nn[i, :, :, 0]))
    nrmse_NN_GS.append(NRMSE(arr_holo1_nn[i, :, :, 0], arr_holo1[i, :, :, 0]))
    
    #left holo
    corr_leftholo_GS.append(np.corrcoef(arr_holo2[i, :, :, 0].flatten(), arr_holo1[i, :, :, 0].flatten())[0, 1])
    corr_leftholo_NN.append(np.corrcoef(arr_holo2[i, :, :, 0].flatten(), arr_holo1_nn[i, :, :, 0].flatten())[0, 1])
    corr_leftholo_phase.append(np.corrcoef(arr_holo2[i, :, :, 0].flatten(), phase_arr[i, :, :, 0].flatten())[0, 1])

    nrmse_leftholo_GS.append(NRMSE(arr_holo2[i, :, :, 0], arr_holo1[i, :, :, 0]))
    nrmse_leftholo_NN .append(NRMSE(arr_holo2[i, :, :, 0], arr_holo1_nn[i, :, :, 0]))
    nrmse_leftholo_phase.append(NRMSE(arr_holo2[i, :, :, 0], phase_arr[i, :, :, 0]))

    #hack GS
    GShack_corr.append(np.corrcoef(arr_GShack[i].flatten(), phase_arr[i, :, :, 0].flatten())[0, 1])
    GShack_nrmse.append(NRMSE(arr_GShack[i], phase_arr[i, :, :, 0]))

#right holo for graphs
mean1_nrmse = np.mean(nrmse_phase_GS)
mean2_nrmse = np.mean(nrmse_phase_NN)
mean3_nrmse = np.mean(nrmse_NN_GS)
sigma1_nrmse = np.std(nrmse_phase_GS)
sigma2_nrmse = np.std(nrmse_phase_NN)
sigma3_nrmse = np.std(nrmse_NN_GS)

mean1_corr = np.mean(corr_phase_GS)
mean2_corr = np.mean(corr_phase_NN)
mean3_corr = np.mean(corr_NN_GS)
sigma1_corr = np.std(corr_phase_GS)
sigma2_corr = np.std(corr_phase_NN)
sigma3_corr = np.std(corr_NN_GS)

#left holo for graphs
meanleftholo1_nrmse = np.mean(nrmse_leftholo_GS)
meanleftholo2_nrmse = np.mean(nrmse_leftholo_NN)
meanleftholo3_nrmse = np.mean(nrmse_leftholo_phase)
sigmaleftholo1_nrmse = np.std(nrmse_leftholo_GS)
sigmaleftholo2_nrmse = np.std(nrmse_leftholo_NN)
sigmaleftholo3_nrmse = np.std(nrmse_leftholo_phase)

meanleftholo1_corr = np.mean(corr_leftholo_GS)
meanleftholo2_corr = np.mean(corr_leftholo_NN)
meanleftholo3_corr = np.mean(corr_leftholo_phase)
sigmaleftholo1_corr = np.std(corr_leftholo_GS)
sigmaleftholo2_corr = np.std(corr_leftholo_NN)
sigmaleftholo3_corr = np.std(corr_leftholo_phase)

#hack for graphs
meanGShack_nrmse = np.mean(GShack_nrmse)
sigmaGShack_nrmse = np.std(GShack_nrmse)
meanGShack_corr = np.mean(GShack_corr)
sigmaGShack_corr = np.std(GShack_corr)

folders = {0: 'fullsize', 1: 'forth_edge', 2: 'forth_center'}
path = f'./graphs/dataset_classifier/{folders[0]}'

# right holo plot 
plot(nrmse_phase_GS, mean1_nrmse, sigma1_nrmse, f"{path}/nrmse_phase_GS")
plot(nrmse_phase_NN, mean2_nrmse, sigma2_nrmse, f"{path}/nrmse_phase_NN")
plot(nrmse_NN_GS, mean3_nrmse, sigma3_nrmse, f"{path}/nrmse_NN_GS")
plot(corr_phase_GS, mean1_corr, sigma1_corr, f"{path}/corr_phase_GS")
plot(corr_phase_NN, mean2_corr, sigma2_corr, f"{path}/corr_phase_NN")
plot(corr_NN_GS, mean3_corr, sigma3_corr, f"{path}/corr_NN_GS")

# left holo plot
plot(nrmse_leftholo_GS, meanleftholo1_nrmse, sigmaleftholo1_nrmse, f"{path}/nrmse_leftholo_GS")
plot(nrmse_leftholo_NN, meanleftholo2_nrmse, sigmaleftholo2_nrmse, f"{path}/nrmse_leftholo_NN")
plot(nrmse_leftholo_phase, meanleftholo3_nrmse, sigmaleftholo3_nrmse, f"{path}/nrmse_leftholo_phase")
plot(corr_leftholo_GS, meanleftholo1_corr, sigmaleftholo1_corr, f"{path}/corr_leftholo_GS")
plot(corr_leftholo_NN, meanleftholo2_corr, sigmaleftholo2_corr, f"{path}/corr_leftholo_NN")
plot(corr_leftholo_phase, meanleftholo3_corr, sigmaleftholo3_corr, f"{path}/corr_leftholo_phase")

# hack plot
plot(GShack_nrmse, meanGShack_nrmse, sigmaGShack_nrmse, f"{path}/GShack_nrmse")
plot(GShack_corr, meanGShack_corr, sigmaGShack_corr, f"{path}/GShack_corr")

