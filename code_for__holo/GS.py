from PIL import Image
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import cv2

import keras

(train_images, _), (test_images, _) = keras.datasets.mnist.load_data()

class GS:
    def __init__(self, image, num):
        self.raw_image = np.array(image) # изображение в numpy
        self.width, self.height, self.length = self.raw_image.shape[0], self.raw_image.shape[1], self.raw_image.shape[2] # ширина & высота
        self.amplitude = self.norm_amplitude() 
        self.phase = 2 * np.pi * np.random.rand(self.width, self.height, self.length) # Инициализация фазы для голограммы
        self.num = num

        # Начальная комплексная амплитуда
        self.complex_amplitude = self.amplitude * np.exp(1j * self.phase) # Создаёт комплексное значение амплитуды, умножая нормированную амплитуду на экспоненту с мнимой частью, равной фазе
        self.SSIM = None # переменная для хранения SSIM (если не проинициализировать, возникнет ошибка, тк класс?)

        self.phase_result = None # переменная для хранения фазы полученной голограммы
        self.result = None # переменную для хранения генерируемого изображения

    def norm_amplitude(self):
        return (self.raw_image / np.max(self.raw_image))**(0.5) # Нормировака амплитуды

    def train(self, epoch=100):
        self.SSIM = np.zeros((7,epoch)) 
        for i in tqdm(range(epoch)): # Создает шкалу прогресса как при обучениии в tensorflow
            # Обратное преобразование Фурье 
            freq_img = fftshift(ifft2(ifftshift(self.complex_amplitude, axes=(1,2)), axes=(1,2)), axes=(1,2)) 
           
            f_img_phase = np.angle(freq_img) #Извлекает фазу из пространственного изображения, angle -> returns an angle of complex value
            f_img_norm = 1 * np.exp(1j * f_img_phase) #Создаёт новое комплексное значение, используя нормированную амплитуду и полученную фазу
            
            # Прямое преобразование Фурье
            space_img = fftshift(ifft2(ifftshift(f_img_norm, axes=(1,2)), axes=(1,2)), axes=(1,2))

            for j in range(7):
                self.SSIM[j][i] = ssim((np.abs((space_img[j]) / np.max(np.abs(space_img[j]))))**2, (np.abs(self.amplitude[j]))**2, data_range=1) # data range -> [0; 1] this value? -> max

            self.complex_amplitude = np.abs(self.amplitude) * np.exp(1j * np.angle(space_img)) # Обновляет комплексное значение амплитуды, используя нормированную амплитуду и фазу из полученного изображения
        

        self.phase_result = np.uint8((f_img_phase + np.pi) / (2*np.pi) * 255) # полученную фазу
        
        # print(np.min(f_img_phase))
        # print(f_img_phase) # отсюда минимум -> число пи
        
        #self.phase_result = np.abs(f_img_phase)

        self.result = np.abs((space_img)**2) # полученное изображение ч/з ГС
        
        #np.savez(f"./{num}", self.phase_result[3]) # used for creating validation_dataset

        cv2.imwrite("geeks.tiff", self.phase_result[3])

        #return self.phase_result, self.format_image(self.amplitude), self.SSIM

        plt.figure(0)
        plt.title("Initial picture")
        plt.imshow(self.raw_image[3][:][:], cmap="gray")

        # Распределение фазы
        plt.figure(1)
        plt.title("Phase result")
        plt.imshow(self.phase_result[3][:][:], cmap="gray")

        # Результат ГС
        plt.figure(2)
        plt.title("Final picture")
        plt.imshow(self.format_image((self.result[3][:][:])), cmap="gray")

        plt.figure(3)
        plt.title("SSIM")
        plt.plot(list(range(epoch)), self.SSIM[3]) # график изменения SSIM по эпохам

        plt.figure(4)
        plt.title("reconstructed image")
        plt.imshow(np.abs(fftshift(ifft2(ifftshift(np.exp(1j*f_img_phase[3]))))), cmap='gray') # график изменения SSIM по эпохам

        plt.show()

    def format_image(self, img):
        img = img * 255 / np.max(img) # -> from [0; 1] --> [0, 255]
        img = img.astype(np.uint8) # Эта строка преобразует тип данных изображения в 8-битный беззнаковый целочисленный (unsigned integer).
        return img
    # В этой строке кода происходит преобразование изображения из формата с плавающей точкой (float) в 8-битный формат (uint8). 
    # Это делается для того, чтобы изображение можно было отобразить или сохранить в стандартном формате, который поддерживает большинство инструментов работы с изображениями.

if __name__ == "__main__":
    # image_path = "./Baboo.tiff"
    # g = Image.open(image_path, mode="r")
    # # convert RGB image to Gray image
    # g = g.convert("L")
    # # Resize
    # g = g.resize((512, 512), Image.BILINEAR) #Returns a resized copy of this image. 
    # # G = np.array(g)
    # gs = GS(g)
    # gs.train()

    mas2 = np.load("./test_images_resized.npz")

    mas2 = mas2["arr_0"]

    gs = GS(mas2[:7], 0)
    gs.train()

    img = cv2.imread('geeks.tiff', cv2.IMREAD_GRAYSCALE)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    a = np.abs(fftshift(ifft2(ifftshift(np.exp(1j*img / 255 * 2 * np.pi)))))
    ax.imshow((a/np.max(a)*255), cmap="gray")

    plt.show()

    # phase, img, Ssim = imG 

    # new_im = fftshift(fft2(fftshift(1*np.exp(1j*phase)), axes=(1,2)), axes=(1,2))

    # for i in range(7):
    #     image11 = GS.format_image(gs, np.abs(new_im[i]))
    #     im = Image.fromarray(image11)
    #     im.save(f"Nimg{i}.jpeg")

    # print(Ssim)
    # np.savetxt("SSim_28.csv", Ssim, delimiter=",")

    # for i in range(7):
    #     im = Image.fromarray(img[i])
    #     im.save(f"img{i}.jpeg")

    # gs2 = GS(train_images[:7], 0)
    # gs2.train()