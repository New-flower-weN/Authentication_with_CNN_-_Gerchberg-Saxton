# Imports
import numpy as np
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

import tensorflow as tf
from tensorflow import keras

# from holography import func
import cv2

NUM = 6

def NRMSE(x,y):
    mse = np.sum((x - y)**2)
    sum = np.sum(x**2)
    nrmse = np.sqrt(mse/sum)
    return nrmse

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

N = 64
N1 = N - 1
ps = 9.0e-6
lmbd = 532.0e-9
z = (N * ps**2) / lmbd 
# z = 0.04
du = lmbd * z / (N * ps)
k = 2 * np.pi / lmbd

X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
# X, Y = np.meshgrid(np.linspace(-N1 / 2, N1 / 2, 64), np.linspace(-N1 / 2, N1 / 2, 64))
du = lmbd * z / (N * ps)
u = ps * X
v = ps * Y

x = du * X
y = du * Y

wave_front = 1j * np.pi * (u**2 + v**2) / (lmbd * z)

Num_of_pics = 1000

arr_img = np.zeros((Num_of_pics, 64, 64))
arr_holo = np.zeros((Num_of_pics, 64, 64))
j = 0

# phase = cv2.imread('./dataset_64/phase_2.png', cv2.IMREAD_GRAYSCALE) / 255.0
phase = np.random.uniform(-1, 1, (64, 64))

print(phase)

for i in range(164001, 164000 + Num_of_pics + 1):
    # img = cv2.imread(f"./dataset_512/pair_{i}/num_{i}.jpeg", cv2.IMREAD_GRAYSCALE)
    # holo = cv2.imread(f"./dataset_512/pair_{i}/holo_{i}.jpeg", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(f'./dataset_64/img/{i}.png', cv2.IMREAD_GRAYSCALE)
    # holo = cv2.imread(f"C:\\Users\\Proku\\OneDrive\\Desktop\\kinoform\\H_k_64_50_3\\K\\{i}.png", cv2.IMREAD_GRAYSCALE)
    holo = cv2.imread(f".\\dataset_64\\holo1\\{i}.png", cv2.IMREAD_GRAYSCALE)

    arr_img[j] = img / 255.0
    arr_holo[j] = holo / 255.0
    j+=1

model = keras.models.load_model("./saved_NN/1phase.h5", compile=False)
# model.compile(optimizer="adam", loss="mean_squared_error", metrics=["root_mean_squared_error"])
model.load_weights("./saved_NN/1phase.weights.h5")

model2 = keras.models.load_model("./saved_NN/2phase.h5", compile=False)
model2.load_weights("./saved_NN/2phase.weights.h5")

model4 = keras.models.load_model("./saved_NN/4phase.h5", compile=False)
model4.load_weights("./saved_NN/4phase.weights.h5")

model8 = keras.models.load_model("./saved_NN/8phase.h5", compile=False)
model8.load_weights("./saved_NN/8phase.weights.h5")

model16 = keras.models.load_model("./saved_NN/16phase.h5", compile=False)
model16.load_weights("./saved_NN/16phase.weights.h5")

arr_img = np.reshape(arr_img, (Num_of_pics, 64, 64, 1))

@tf.function(reduce_retracing=True)
def predict_all(models, arr_img):
    return [m(arr_img, training=False) for m in models]

results = np.array(predict_all([model, model2, model4, model8, model16], arr_img))

result = results[0, :, :, :, 0]
result2 = results[1, :, :, :, 0]
result4 = results[2, :, :, :, 0]
result8 = results[3, :, :, :, 0]
result16 = results[4, :, :, :, 0]

# print(mas2)
print(results.shape)

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# print(result[150])

ax = fig.add_subplot(2, 6, 1)
ax.imshow(np.abs(arr_holo[NUM]), cmap='gray')

ax = fig.add_subplot(2, 6, 7)
II = np.abs(fftshift(fft2(ifftshift(np.exp(1j * arr_holo[NUM] * 2 * np.pi) * np.exp(wave_front)))))
I0 = (II/np.max(II))**2
ax.imshow(I0, cmap='gray')

ax = fig.add_subplot(2, 6, 2)
ax.imshow(np.abs(result[NUM]), cmap="gray")

ax = fig.add_subplot(2, 6, 8)
II = np.abs(fftshift(fft2(ifftshift(np.exp(1j * result[NUM] * 2 * np.pi) * np.exp(wave_front)))))
I1 = (II/np.max(II))**2
ax.imshow(I1, cmap='gray')

ax = fig.add_subplot(2, 6, 3)
ax.imshow(np.abs(result2[NUM]), cmap="gray")

ax = fig.add_subplot(2, 6, 9)
II = np.abs(fftshift(fft2(ifftshift(np.exp(1j * result2[NUM] * 2 * np.pi) * np.exp(wave_front)))))
I2 = (II/np.max(II))**2
ax.imshow(I2, cmap='gray')

ax = fig.add_subplot(2, 6, 4)
ax.imshow(np.abs(result4[NUM]), cmap="gray")

ax = fig.add_subplot(2, 6, 10)
II = np.abs(fftshift(fft2(ifftshift(np.exp(1j * result4[NUM] * 2 * np.pi) * np.exp(wave_front)))))
I4 = (II/np.max(II))**2
ax.imshow(I4, cmap='gray')

ax = fig.add_subplot(2, 6, 5)
ax.imshow(np.abs(result8[NUM]), cmap="gray")

ax = fig.add_subplot(2, 6, 11)
II = np.abs(fftshift(fft2(ifftshift(np.exp(1j * result8[NUM] * 2 * np.pi) * np.exp(wave_front)))))
I8 = (II/np.max(II))**2
ax.imshow(I8, cmap='gray')

ax = fig.add_subplot(2, 6, 6)
ax.imshow(np.abs(result16[NUM]), cmap="gray")

ax = fig.add_subplot(2, 6, 12)
II = np.abs(fftshift(fft2(ifftshift(np.exp(1j * result16[NUM] * 2 * np.pi) * np.exp(wave_front)))))
I16 = (II/np.max(II))**2
ax.imshow(I16, cmap='gray')

plt.show()

corr_list1 = []
corr_list2 = []
corr_list3 = []
corr_list4 = []
corr_list5 = []
corr_list6 = []

print(np.array(arr_holo).shape)

# Рассчет коэффициента корреляции НС голограмм и исх. фазового распределения, но 1 для самого ГС
for i in range(Num_of_pics):
    corr_list1.append(np.corrcoef(arr_holo[i].flatten(), phase.flatten())[0, 1])
    corr_list2.append(np.corrcoef(result[i].flatten(), phase.flatten())[0, 1])
    corr_list3.append(np.corrcoef(result2[i].flatten(), phase.flatten())[0, 1])
    corr_list4.append(np.corrcoef(result4[i].flatten(), phase.flatten())[0, 1])
    corr_list5.append(np.corrcoef(result8[i].flatten(), phase.flatten())[0, 1])
    corr_list6.append(np.corrcoef(result16[i].flatten(), phase.flatten())[0, 1])

# print(corr_list1)
# Print correlation results as a formatted table
headers = ["--------", "GS", "1-phase", "2-phase", "4-phase", "8-phase", "16-phase"]
data = [
    ["mean corr"] + [np.mean(corr_list1), np.mean(corr_list2), np.mean(corr_list3), 
     np.mean(corr_list4), np.mean(corr_list5), np.mean(corr_list6)],
    ["max corr"] + [np.max(corr_list1), np.max(corr_list2), np.max(corr_list3),
     np.max(corr_list4), np.max(corr_list5), np.max(corr_list6)],
    ["min corr"] + [np.min(corr_list1), np.min(corr_list2), np.min(corr_list3),
     np.min(corr_list4), np.min(corr_list5), np.min(corr_list6)]
]

# Format the table with aligned columns
row_format = "{:<10}" + "{:<15}" * 6
print(row_format.format(*headers))
for row in data:
    # Format numbers to 4 decimal places
    formatted_row = [row[0]] + [f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x) for x in row[1:]]
    print(row_format.format(*formatted_row))

#print(NRMSE(I1, I2), NRMSE(I1, I3))
#print(np.corrcoef(I1.flatten(), I2.flatten())[0, 1], np.corrcoef(I1.flatten(), I3.flatten())[0, 1])

# # ниже для голограммы Френеля

# # Восстановление голограммы

# im = np.load("./resized_img_128.npz")
# im = im['arr_0'][:10] / 255

# model = keras.models.load_model("./UNetW.h5", compile=False)
# model.load_weights("./UNetW.weights.h5")

# result = model.predict(im)

# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)

# ax = fig.add_subplot(2, 2, 1)
# ax.imshow(im[NUM], cmap="gray")

# ax = fig.add_subplot(2, 2, 2)
# ax.imshow(result[NUM], cmap="gray")

# #### NN reconstruction

# c3 = np.exp(1j * k * z) / (1j * lmbd * z) * np.exp(1j * np.pi * (x ** 2 + y ** 2) / (lmbd * z))
# #reconstructed_field = c3 * fftshift(ifft2(ifftshift(I* opor * np.exp(1j * np.pi * (u ** 2 + v ** 2) / (lmbd * z)))))
# reconstructed_field = c3 * fftshift(ifft2(ifftshift(result[:,:,:,0] * np.exp(1j * np.pi * (u ** 2 + v ** 2) / (lmbd * z)), axes=(1,2)), axes=(1,2)), axes=(1,2)) # opor не влияет на восстановление

# ax = fig.add_subplot(2, 2, 3)
# ax.imshow(np.abs(reconstructed_field[NUM]), cmap="gray")

# plt.show()

# # for i in range(10):
# #     cv2.imwrite(f'./folder_rubbish/_{i}.jpeg', np.uint8(im[i]/np.max(im[i]) * 255))
# #     cv2.imwrite(f'./folder_rubbish/__{i}.jpeg', np.uint8(result[i]/np.max(result[i]) * 255))
# #     cv2.imwrite(f'./folder_rubbish/___{i}.jpeg', np.uint8(np.abs(reconstructed_field[i])/np.max(np.abs(reconstructed_field[i])) * 255))

# """               DO NOT GO ANY FURTHER                  """

# images_2 = np.abs(reconstructed_field)[:,(128-51):,(128-51):]

# for i in range(10):    
#     images_2[i] = cv2.rotate(images_2[i]/np.max(images_2[i]), cv2.ROTATE_180)

# images_1 = [cv2.imread(f'_{i}.jpeg', cv2.IMREAD_GRAYSCALE)[:51, :51]/255 for i in range(10)]
# # images_2 = [cv2.imread(f'___{i}.jpeg', cv2.IMREAD_GRAYSCALE)[:51, :51] for i in range(10)] 
# cv2.imshow('dasf1', images_1[0])
# cv2.imshow('dasf2', images_2[0])
# cv2.waitKey(0)
# arr_ssim = []

# for i in range(10):
#     arr_ssim.append(ssim(images_1[i], images_2[i], data_range=1, win_size=51))

# print(arr_ssim)