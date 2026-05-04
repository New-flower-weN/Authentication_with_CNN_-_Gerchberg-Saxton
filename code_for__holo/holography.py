from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import cv2
from tqdm import tqdm

# import keras

# (train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
# train_images = np.load("./train_images_resized.npz")
# train_images = np.load("./resized_img_128.npz")
# train_images = train_images["arr_0"]

def fresnel(picture=None, num=0):
    N = 1024
    ps = 8.0e-6
    lmbd = 532.0e-9
    z = 0.2 #(N * ps**2) / lmbd 
    # z = 0.04
    du = lmbd * z / (N * ps)

    # x1 = np.linspace(-N//2, N//2, N) #Координата в поле объекта x
    # y1 = np.linspace(-N//2, N//2, N) #Координата в поле объекта y
    # u1 = np.linspace(-N//2, N//2, N) #Координата в поле восстановления u 
    # v1 = np.linspace(-N//2, N//2, N) #Координата в поле восстановления v

    # x, y = np.meshgrid(x1, y1)
    # u, v = np.meshgrid(u1, v1)

    # x,y = x*du, y*du
    # u,v = u*ps, v*ps

    # print(x, len(x))

    """"""""""""""""""""""""""""""""""""""""""

    X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
    du = lmbd * z / (N * ps) # why
    u = ps * X
    v = ps * Y

    x = du * X
    y = du * Y

    # print(x, len(x))

    """"""""""""""""""""""""""""""""""""""""""

    k = 2 * np.pi / lmbd

    print(x.shape[0], x.shape[1])

    # object = cv2.imread("./Baboo.tiff", cv2.IMREAD_GRAYSCALE)
    # object = np.reshape(object, (1, 512, 512))
    object = picture

    # object = train_images
    # object = object["arr_0"][7500]
    # object_field = np.sqrt(np.array(object) / 255)

    # Для разделения на +-1 и 0 порядки
    depth, height, width = object.shape[:3]

    c_obj = np.empty([depth, 205, 205]) # = 205 -> 2 и 3

    print(c_obj.shape)

    # look later if there's better way of resizing
    count = 0
    for im in object:
        small_im = cv2.resize(im, (205, 205), interpolation=cv2.INTER_AREA) # = 205 for 512x512 pix
        c_obj[count] = small_im 
        count+=1

    #small_image = cv2.resize(object, (width//2, height//2), interpolation=cv2.INTER_AREA)
    final_image = np.zeros((depth, height, width), dtype=np.uint8)
    final_image[:c_obj.shape[0], :c_obj.shape[1], :c_obj.shape[2]] = c_obj
    object_field = np.sqrt(np.array(final_image) / np.max(final_image)) 

    c1 = np.exp(1j * k * z) / (1j * lmbd * z) * np.exp(1j * k * (x ** 2 + y ** 2) / (2 * z))
    c2 = object_field * np.exp(1j * k * (u ** 2 + v ** 2) / (2 * z)) # ТУТ МОМЕНТ ПОСМОТРЕТЬ С УМНОЖЕНИЕМ # ПРОВЕРЕНО, ВСЕ ОК

    obj = c1 * fftshift(fft2(ifftshift(c2, axes=(1,2)), axes=(1,2)), axes=(1,2)) # тут тоже самое # ПРОВЕРЕНО, ВСЕ ОК
    #obj = c1 * fftshift(fft2(ifftshift(c2)))

    ref = np.sqrt(np.mean(np.abs(obj) ** 2))
    opor = ref #* np.ones((512, 512))
    obj += opor

    I = np.abs(obj) ** 2 

    #show graphs & stuff

    # plt.imshow(final_image[0], cmap='gray')
    # plt.show()

    # plt.imshow(I[0], cmap='gray')
    # plt.show()

    # np.savez("fresnel_holo_train", I)

    # Восстановление голограммы

    c3 = np.exp(1j * k * z) / (1j * lmbd * z) * np.exp(1j * k * (u ** 2 + v ** 2) / (2 * z)) # В ВОССТАНОВЛЕНИИ ТО ЖЕ САМОЕ # ПРОВЕРИЛ, ВСЕ ОК
    #reconstructed_field = c3 * fftshift(ifft2(ifftshift(I* opor * np.exp(1j * np.pi * (u ** 2 + v ** 2) / (lmbd * z)))))
    reconstructed_field = c3 * fftshift(ifft2(ifftshift(I * np.exp(1j * k * (x ** 2 + y ** 2) / (2 * z)), axes=(1,2)), axes=(1,2)), axes=(1,2)) # opor не влияет на восстановление

    # Вывод восстановленного изображения
    reconstructed_intensity = np.abs(reconstructed_field) # тут без квадрата, т.к. иначе все тусклое будет 
    # plt.imshow(reconstructed_intensity[0], cmap='gray')
    # plt.title('Восстановленное изображение')
    # plt.show()

    return I, final_image, reconstructed_field

    # np.savez(f"test_fresnel_128_{num}", I)

# print(train_images[:5].shape)

# for i in range(2):
#     func(train_images[10000*i:10000*(i+1)], i)

# cv2.imwrite("1.jpeg",train_images[8])

# mas = [0]*256

# for i in range(256):
#     for j in X:
#         for k in j:
#             if k == i:
#                 mas[i] +=1
# print(mas)

def fresnel_one(picture, z, pos=0):
    N = 1200
    ps = 7.8e-6
    lmbd = 561.0e-9
    # z = (N * ps**2) / lmbd 
    # print(z)
    # z = 0.04
    du = lmbd * z / (N * ps)

    X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
    du = lmbd * z / (N * ps) # why?
    u = ps * X
    v = ps * Y

    x = du * X
    y = du * Y

    k = 2 * np.pi / lmbd

    # object = picture 

    # Для разделения на +-1 и 0 порядки
    # height, width = (1024, 1024)

    # small_im = cv2.resize(object, (256, 256), interpolation=cv2.INTER_AREA) # = 205 for 512x512 pix
    # small_im = small_im 


    # #small_image = cv2.resize(object, (width//2, height//2), interpolation=cv2.INTER_AREA)
    # final_image = np.zeros((height, width), dtype=np.uint8)
    # if pos == 0:
    #     final_image[:small_im.shape[0], :small_im.shape[1]] = small_im
    # elif pos == 1:
    #     final_image[1024-small_im.shape[0]:, 1024-small_im.shape[1]:] = small_im
    # else:
    #     final_image[:small_im.shape[0]:, 1024-small_im.shape[1]:] = small_im
    
    phase_mask = cv2.imread('./dataset_32/phase_1024_05.png', cv2.IMREAD_GRAYSCALE) / 255.0 * 2 * np.pi - np.pi # 2 * np.pi * (np.random.rand(N, N) - 0.5)
    object_field = np.sqrt(np.array(picture) / np.max(picture)) * np.exp(1j * phase_mask)

    c1 = np.exp(1j * k * z) / (1j * lmbd * z) * np.exp(1j * np.pi * (u ** 2 + v ** 2) / (lmbd * z))
    c2 = object_field * np.exp(1j * np.pi * (x ** 2 + y ** 2) / (lmbd * z)) # ТУТ МОМЕНТ ПОСМОТРЕТЬ С УМНОЖЕНИЕМ # ПРОВЕРЕНО, ВСЕ ОК

    obj = c1 * fftshift(fft2(ifftshift(c2))) # тут тоже самое # ПРОВЕРЕНО, ВСЕ ОК
    #obj = c1 * fftshift(fft2(ifftshift(c2)))

    ref = np.sqrt(np.mean(np.abs(obj) ** 2))
    opor = ref #np.ones((1024, 1024))
    # obj += opor

    # I = np.abs(obj) ** 2 
    I = 2*np.real(opor*np.conj(obj)) 
    I_min = np.min(I)
    I = I - I_min

    # Восстановление голограммы
    z = -z

    c3 = np.exp(1j * k * z) / (1j * lmbd * z) * np.exp(1j * np.pi * (x ** 2 + y ** 2) / (lmbd * z)) # В ВОССТАНОВЛЕНИИ ТО ЖЕ САМОЕ # ПРОВЕРИЛ, ВСЕ ОК
    #reconstructed_field = c3 * fftshift(ifft2(ifftshift(I* opor * np.exp(1j * np.pi * (x ** 2 + y ** 2) / (lmbd * z)))))
    reconstructed_field = c3 * ifftshift(ifft2(fftshift(I * np.exp(1j * np.pi * (u ** 2 + v ** 2) / (lmbd * z))))) # opor не влияет на восстановление

    # Вывод восстановленного изображения
    reconstructed_intensity = np.abs(reconstructed_field) # тут без квадрата, т.к. иначе все тусклое будет 

    return picture, I, reconstructed_intensity

def check():
    in_img = np.zeros((10, 512, 512))
    i = 0
    NUM=3

    for j in tqdm(range(10)):
        for l in range(j, 10+j):
            in_img[i] = cv2.imread(f'./numbers/{l}.tiff', cv2.IMREAD_GRAYSCALE) #/ 255.0
        i+=1

    holo, img, reconstructed = fresnel(in_img)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(in_img[NUM], cmap="gray")

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(np.abs(img[NUM]), cmap="gray")

    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(np.abs(reconstructed[NUM])**2, cmap="gray")

    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(np.abs(holo[NUM]), cmap="gray")

    plt.show()

    for i in range(10):
        cv2.imwrite(f'1_for_preseint_{i}.jpeg', np.uint8(in_img[i]*255))
        cv2.imwrite(f'2_for_preseint_{i}.jpeg', np.uint8(holo[i] / np.max(holo[i]) * 255))

        var = np.abs(reconstructed)[i]    
        cv2.imwrite(f'3_for_preseint_{i}.jpeg', np.uint8(var / np.max(var) * 255))

if __name__ == "__main__":

    for i in tqdm(range(100001, 170000)):

        img = cv2.imread(f"dataset_32/img_angled_1200/{i}.png", cv2.IMREAD_GRAYSCALE)

        IMG, holo, res = fresnel_one(img, 0.4, 0)

        # plt.imshow(IMG, cmap='gray')
        # plt.show()

        # plt.imshow(holo, cmap='gray')
        # plt.show()

        # print(np.corrcoef(holo.flatten(), np.uint8(holo/np.max(holo)*255).flatten()))

        # plt.imshow(res, cmap='gray')
        # plt.show()

        cv2.imwrite(f"dataset_32/fresnel_of_angled_img/{i}.png", np.uint8(holo/np.max(holo)*255))

    # for i in range(10):
    #     cv2.imwrite(f'_{i}.jpeg', np.uint8((np.abs(sth[i])/np.max(np.abs(sth[i])))*255))
    #     cv2.imwrite(f'__{i}.jpeg', np.uint8((np.abs(obj[i])/np.max(np.abs(obj[i])))*255))
    #     cv2.imwrite(f'___{i}.jpeg', np.uint8((np.abs(reconstructed_field[i])/np.max(np.abs(reconstructed_field[i])))*255))



"""            НИЖЕ ПРО 3 ИЗОБРАЖЕНИЯ В РАЗНЫХ УГЛАХ                 """

    # arr = np.zeros((3, 512, 512))
    # holo_arr = np.zeros((3, 512, 512))
    # hologram = np.zeros((512, 512))
    # img1 = cv2.imread('Baboo.tiff', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    # img3 = cv2.imread('spoodiman.jpg', cv2.IMREAD_GRAYSCALE)
    # arr[0] = img1
    # arr[1] = img2
    # arr[2] = img3
    # z = [0.2, 0.25, 0.3]
    # for i in range(3):
    #     holo_arr[i] = fresnel_one(arr[i], z[i], i)[0]

    # hologram = holo_arr[0] + holo_arr[1] + 0*holo_arr[2]

    # plt.imshow(hologram, cmap='gray')
    # plt.show()
    # z = 0.25
    # c3 = np.exp(1j * k * z) / (1j * lmbd * z) * np.exp(1j * np.pi * (x ** 2 + y ** 2) / (lmbd * z)) # В ВОССТАНОВЛЕНИИ ТО ЖЕ САМОЕ # ПРОВЕРИЛ, ВСЕ ОК
    # reconstructed_field = c3 * fftshift(ifft2(ifftshift(hologram * np.exp(1j * np.pi * (u ** 2 + v ** 2) / (lmbd * z))))) # opor не влияет на восстановление

    # plt.imshow(np.abs(reconstructed_field), cmap='gray')
    # plt.show()

    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # ax = fig.add_subplot(2, 2, 1)
    # ax.imshow((np.abs(reconstructed_field)[:154, :154]), cmap="gray")

    # ax = fig.add_subplot(2, 2, 2)
    # ax.imshow(np.abs(reconstructed_field)[512-154:, :154], cmap="gray")

    # ax = fig.add_subplot(2, 2, 3)
    # ax.imshow(np.abs(reconstructed_field)[512-154:, 512-154:], cmap="gray")

    # plt.show()