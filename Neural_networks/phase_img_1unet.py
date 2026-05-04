import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
import random
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# setting physical parameters
N = 32
N1 = N - 1
ps = 8.0e-6
lmbd = 633.0e-9
BATCH_SIZE = 64
epochs = 100
initial_alpha = 0.0
# Глобальная переменная для alpha
current_alpha = tf.Variable(initial_alpha, trainable=False)
final_alpha = 1.0
total_epochs = 20
z = 0.2 # np.sqrt(N * N) * ps**2 / lmbd 
du = lmbd * z / (N * ps)

# Creating grid for fourier transformation
X, Y = tf.meshgrid(
    tf.range(-N // 2, N // 2, dtype=tf.float32),
    tf.range(-N // 2, N // 2, dtype=tf.float32))

X = tf.cast(X, tf.complex64)
Y = tf.cast(Y, tf.complex64)

du = lmbd * z / (N * ps)
u = ps * X
v = ps * Y
x = du * X
y = du * Y

# for shortening creating var that'll be under exp()
wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)
wave_front = tf.cast(wave_front, tf.complex64)

def visualize_fft_results(y_true, y_pred, sample_index=0):
    """
    Визуализация FFT результатов для одного примера из батча (img), (img, holo) v1
    """
    # Выбираем конкретный пример из батча
    y_true_single_img = y_true[sample_index, :, :, 0]
    # y_true_sigle_phase = y_true[sample_index, :, :, 2]
    y_true_sigle_holo = y_true[sample_index, :, :, 1]
    y_pred_single = y_pred[sample_index, :, :, 0]
    
    # Убираем последнюю ось (каналы), если она равна 1
    y_true_2d_img  = tf.squeeze(y_true_single_img)
    y_true_2d_holo = tf.squeeze(y_true_sigle_holo)
    y_pred_2d = tf.squeeze(y_pred_single)

    y_true_2d_holo = tf.cast(y_true_2d_holo, tf.complex64)
    y_pred_2d = tf.cast(y_pred_2d, tf.complex64)

    # Compute FFTs
    y_true_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*(y_true_2d_holo*2*np.pi-np.pi)) * tf.exp(wave_front), axes=(0, 1))), axes=(0, 1)) 
    y_pred_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*(y_pred_2d*2*np.pi-np.pi)) * tf.exp(wave_front), axes=(0, 1))), axes=(0, 1)) 

    # Calculate magnitudes
    y_true_fft_abs = tf.abs(y_true_fft).numpy()
    y_pred_fft_abs = tf.abs(y_pred_fft).numpy()

    # Convert back to float for display
    y_true_2d_display = tf.cast(y_true_2d_holo, tf.float32).numpy()
    y_pred_2d_display = tf.cast(y_pred_2d, tf.float32).numpy()

    """
    Визуализация FFT результатов для одного примера из батча (img), (img, holo) v1 2 predict pics
    """
    # # Выбираем конкретный пример из батча
    # y_true_single_img = y_true[sample_index, :, :, 0]
    # y_true_sigle_holo = y_true[sample_index, :, :, 1]
    # y_pred_single1 = y_pred[sample_index, :, :, 0]
    # y_pred_single2 = y_pred[sample_index, :, :, 1]
    
    # # Убираем последнюю ось (каналы), если она равна 1
    # y_true_2d_img  = tf.squeeze(y_true_single_img)
    # y_true_2d_holo = tf.squeeze(y_true_sigle_holo)
    # y_pred_2d1 = tf.squeeze(y_pred_single1)
    # y_pred_2d2 = tf.squeeze(y_pred_single2)

    # y_true_2d_holo = tf.cast(y_true_2d_holo, tf.complex64)
    # y_pred_2d = tf.cast(y_pred_2d1, tf.complex64)

    # # Compute FFTs
    # y_true_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*(y_true_2d_holo*2*np.pi-np.pi)) * tf.exp(wave_front), axes=(0, 1))), axes=(0, 1)) 
    # y_pred_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*(y_pred_2d*2*np.pi-np.pi)) * tf.exp(wave_front), axes=(0, 1))), axes=(0, 1)) 

    # # Calculate magnitudes
    # y_true_fft_abs = tf.abs(y_true_fft).numpy()
    # y_pred_fft_abs = tf.abs(y_pred_fft).numpy()

    # # Convert back to float for display
    # y_true_2d_display = tf.cast(y_true_2d_holo, tf.float32).numpy()
    # y_pred_2d_display = tf.cast(y_pred_2d, tf.float32).numpy()

    """
    Визуализация FFT результатов для одного примера из батча (img), (img, holo) v2
    """
    # # Выбираем конкретный пример из батча
    # y_true_single_img = y_true[sample_index, :, :, 0]
    # y_true_sigle_holo = y_true[sample_index, :, :, 1]
    # y_pred_single = y_pred[sample_index]
    
    # # Убираем последнюю ось (каналы), если она равна 1
    # y_true_2d_img  = tf.squeeze(y_true_single_img)
    # y_true_2d_holo = tf.squeeze(y_true_sigle_holo)
    # y_pred_2d = tf.squeeze(y_pred_single)

    # y_true_2d_holo = tf.cast(y_true_2d_holo, tf.complex64)
    # amplitude = tf.sqrt(tf.cast(y_true_2d_img, tf.complex64))
    # y_pred_2d = tf.cast(y_pred_2d, tf.complex64)

    # # Compute IFFTs
    # y_pred_fft = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.fftshift(amplitude * tf.exp(1j*(y_pred_2d*2*np.pi-np.pi)), axes=(0, 1))), axes=(0, 1)) * tf.exp(-wave_front)
    # y_pred_holo = tf.math.angle(y_pred_fft)
    # y_pred_holo = tf.cast(y_pred_holo, tf.complex64)

    # # Compute FFTs
    # y_true_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*(y_true_2d_holo*2*np.pi-np.pi)) * tf.exp(wave_front), axes=(0, 1))), axes=(0, 1)) 
    # y_pred_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*y_pred_holo) * tf.exp(wave_front), axes=(0, 1))), axes=(0, 1)) 

    # # Calculate magnitudes
    # y_true_fft_abs = tf.abs(y_true_fft).numpy()
    # y_pred_fft_abs = tf.abs(y_pred_fft).numpy()

    # # Convert back to float for display
    # y_true_2d_display = tf.cast(y_true_2d_holo, tf.float32).numpy()
    # y_pred_2d_display = tf.cast(y_pred_holo, tf.float32).numpy()


    '''
    Визуализация FFT результатов для одного примера из батча (img), (holo)
    '''
    # # Выбираем конкретный пример из батча
    # y_true_single = y_true[sample_index]
    # y_pred_single = y_pred[sample_index]
    
    # # Убираем последнюю ось (каналы), если она равна 1
    # y_true_2d = tf.squeeze(y_true_single)
    # y_pred_2d = tf.squeeze(y_pred_single)

    # y_true_2d = tf.cast(y_true_2d, tf.complex64)
    # y_pred_2d = tf.cast(y_pred_2d, tf.complex64)

    # # Compute FFTs
    # y_true_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*(y_true_2d*2*np.pi-np.pi)) * tf.exp(wave_front), axes=(0, 1))), axes=(0, 1)) 
    # y_pred_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*(y_pred_2d*2*np.pi-np.pi)) * tf.exp(wave_front), axes=(0, 1))), axes=(0, 1)) 

    # # Calculate magnitudes
    # y_true_fft_abs = tf.abs(y_true_fft).numpy()
    # y_pred_fft_abs = tf.abs(y_pred_fft).numpy()

    # # Convert back to float for display
    # y_true_2d_display = tf.cast(y_true_2d, tf.float32).numpy()
    # y_pred_2d_display = tf.cast(y_pred_2d, tf.float32).numpy()
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(y_true_2d_display, cmap='gray')
    axes[0, 0].set_title('Original True')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(y_pred_2d_display, cmap='gray')  # Fixed: use y_pred_2d_display instead of y_pred
    axes[0, 1].set_title('Original Pred')
    axes[0, 1].axis('off')
    
    # FFT magnitudes
    axes[1, 0].imshow(y_true_fft_abs**2, cmap='gray')
    axes[1, 0].set_title('True FFT Magnitude')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(y_pred_fft_abs**2, cmap='gray')
    axes[1, 1].set_title('Pred FFT Magnitude')
    axes[1, 1].axis('off')

    # axes[2, 1].imshow(y_true_single_img, cmap='gray')
    # axes[2, 1].set_title('Pred FFT Magnitude')
    # axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # return y_true_single, y_pred_single

def custom_loss(y_true, y_pred):
    y_true_2d_img = tf.squeeze(y_true[:, :, :, 0])
    y_true_2d_holo = tf.squeeze(y_true[:, :, :, 1])
    y_pred_2d1 = tf.squeeze(y_pred[:, :, :, 0]) 
    # y_pred_2d2 = tf.squeeze(y_pred[:, :, :, 1])
    
    # y_true_complex = tf.cast(y_true_2d, tf.complex64)
    y_pred_complex = tf.cast(y_pred_2d1, tf.complex64)
    
    # y_true_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*(y_true_complex*2*np.pi-np.pi)) * np.exp(wave_front), axes=(1, 2))), axes=(1, 2))
    y_pred_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*(y_pred_complex*2*np.pi-np.pi)) * tf.exp(wave_front), axes=(1, 2))), axes=(1, 2))
    
    # spatial_loss1 = tf.reduce_mean(tf.square(y_true - y_pred))
    # spatial_loss2 = tf.reduce_mean(1 - pearson_r(y_true, y_pred)) 
    
    # y_true_fft_abs = tf.square(tf.abs(y_true_fft))
    y_pred_fft_abs = tf.abs(y_pred_fft)
    # max_val_true = tf.reduce_max(y_true_fft_abs, axis=[1, 2], keepdims=True)
    # y_true_fft_abs = y_true_fft_abs / max_val_true
    max_val_pred = tf.reduce_max(y_pred_fft_abs, axis=[1, 2], keepdims=True)
    y_pred_fft_abs = y_pred_fft_abs / max_val_pred
    y_pred_fft_abs = tf.square(y_pred_fft_abs)
    # freq_loss1 = tf.reduce_mean(tf.square(y_true_fft_abs - y_pred_fft_abs)) 
    # freq_loss2 = tf.reduce_mean(1 - ssim_metric(y_true_fft_abs, y_pred_fft_abs)) 
    # freq_loss3 = tf.reduce_mean(1 - pearson_r(y_true_fft_abs, y_pred_fft_abs)) 

    # return 0.9*(1-current_alpha)*spatial_loss1 + 0.1*(1-current_alpha)*spatial_loss2 + 0.8*(current_alpha)*freq_loss1 + 0.2*(current_alpha)*freq_loss3
    # return tf.reduce_mean(1 - ssim_metric(y_pred_fft_abs, y_true_2d_img)) #+ tf.reduce_mean(tf.square(y_pred_2d2 - y_true_2d_holo))
    return tf.reduce_mean(1 - ssim_metric(y_pred_fft_abs, y_true_2d_img)) + tf.reduce_mean(tf.square(pearson_r(y_pred_2d1, y_true_2d_holo) - 0.1))

def custom_loss2(y_true, y_pred):
    y_true_2d_img = tf.squeeze(y_true[:, :, :, 0])
    y_true_2d_holo = tf.squeeze(y_true[:, :, :, 1])
    y_true_2d_phase = tf.squeeze(y_true[:, :, :, 2])
    y_pred_2d1 = tf.squeeze(y_pred[:, :, :, 0])
    # y_pred_2d2 = tf.squeeze(y_pred[:, :, :, 1])
    
    # y_true_complex = tf.cast(y_true_2d, tf.complex64)
    y_pred_complex = tf.cast(y_pred_2d1, tf.complex64)
    y_true_2d_phase = tf.cast(y_true_2d_phase, tf.complex64)
    y_true_img = tf.cast(y_true_2d_img, tf.complex64)
    
    y_pred_fft = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.fftshift(tf.sqrt(y_true_img)*tf.exp(1j*(y_true_2d_phase*2*np.pi-np.pi)), axes=(1, 2))), axes=(1, 2)) * tf.exp(-wave_front)
    # y_pred_fft = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.fftshift(tf.sqrt(y_pred_complex)*tf.exp(1j*(y_true_2d_phase*2*np.pi-np.pi)), axes=(1, 2))), axes=(1, 2)) * tf.exp(-wave_front)
    
    y_pred_fft_abs = tf.math.angle(y_pred_fft) + np.pi
    max_val_pred = tf.reduce_max(y_pred_fft_abs, axis=[1, 2], keepdims=True)
    y_pred_fft_abs = y_pred_fft_abs / max_val_pred

    # loss = tf.reduce_mean(tf.square(tf.abs(y_pred_fft) - tf.abs(y_true_fft))) + tf.reduce_mean(tf.square(tf.math.angle(y_pred_fft) - tf.math.angle(y_true_fft))) + tf.reduce_mean(tf.square(y_pred_2d1 - y_true_2d_img))
    loss = tf.reduce_mean(tf.square(y_pred_fft_abs - y_true_2d_holo))

    # return tf.reduce_mean(tf.square(y_pred_fft_abs - y_true_2d_holo)) + tf.reduce_mean(tf.square(y_pred_2d1 - y_true_2d_img))
    return  loss

# Callback для обновления alpha
class AlphaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_alpha=0.01, final_alpha=1, total_epochs=80):
        super().__init__()
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_epochs = total_epochs
        self.check = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        # # Линейное увеличение alpha
        # if epoch >= 20 and epoch % 5 == 0:
        #     progress = (epoch-20+1) / self.total_epochs
        #     new_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        #     new_alpha = min(new_alpha, self.final_alpha)
            
        #     current_alpha.assign(new_alpha)
        #     print(f"Epoch {epoch+1}: alpha = {new_alpha:.4f}")

        # if epoch % 5 == 0 and self.check == 1:
        #     current_alpha.assign(1.0)
        #     self.check = 0
        # elif epoch % 5 == 0 and self.check == 0:
        #     current_alpha.assign(0.0)
        #     self.check = 1

        if epoch % 3 == 0 and epoch != 0:
            current_alpha.assign(1.0)

class AlphaScheduler2(tf.keras.callbacks.Callback):
    def __init__(self, switch_interval=5, initial_alpha=1.0):
        super().__init__()
        self.switch_interval = switch_interval
        self.current_alpha = initial_alpha
    
    def on_epoch_begin(self, epoch, logs=None):
        # Переключаем alpha каждые switch_interval эпох
        if epoch % self.switch_interval == 0:
            # Меняем alpha на противоположное значение
            new_alpha = 1.0 if self.current_alpha == 0.0 else 0.0
            
            current_alpha.assign(new_alpha)
            self.current_alpha = new_alpha
            print(f"Epoch {epoch+1}: alpha переключено на {new_alpha:.1f}")
        else:
            print(f"Epoch {epoch+1}: alpha остается {self.current_alpha:.1f}")

def load_and_process_image(file_path):
    """
    Load and process a single image file
    num_of_channels parameter is kept for compatibility but not used in indexing
    """
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1, dtype=tf.dtypes.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def create_dataset(img_files, phase_file, holo_files, indices, num_of_folders):
    img_dataset = []
    holo_dataset_true = []
    holo_dataset_zero_corr = []
    phase_dataset = []
    phase_img_plane_dataset = []

    for idx in indices:
        img_dataset.append(img_files[idx])
        phase_dataset.append(phase_file)
        holo_dataset_zero_corr.append(holo_files[0][idx])
        holo_dataset_true.append(holo_files[1][idx])
        phase_img_plane_dataset.append(holo_files[2][idx])

    # Create dataset with tuple structure
    ds = tf.data.Dataset.from_tensor_slices(((img_dataset, phase_dataset), (holo_dataset_true, holo_dataset_zero_corr, phase_img_plane_dataset)))
    
    # Map function that handles the nested structure
    def process_nested_files(file_paths, holo_path):
        img_path, phase_path = file_paths
        holo_true_path, holo_zero_path, phase_img_plane = holo_path
        # img = load_and_process_image(img_path)
        phase = load_and_process_image(phase_path)
        holo = load_and_process_image(holo_true_path)
        zero_holo = load_and_process_image(holo_zero_path)
        # phase_img_plane = load_and_process_image(phase_img_plane)


        # img = tf.reshape(img, (128, 128, 1))
        phase = tf.reshape(phase, (N, N, 1))
        holo = tf.reshape(holo, (N, N, 1))
        zero_holo = tf.reshape(zero_holo, (N, N, 1))
        # phase_img_plane = tf.reshape(phase_img_plane, (128, 128, 1))


        input_tensor = tf.concat([zero_holo, phase], axis=-1)
        # output_tensor = tf.concat([img, holo, phase_img_plane], axis=-1)

        return input_tensor, holo
    
    ds = ds.map(process_nested_files, num_parallel_calls=tf.data.AUTOTUNE)

    return ds

# down_sampling block for NN
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    c = keras.layers.Dropout(0.2)(c)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    p = keras.layers.MaxPool2D(pool_size=2, strides=None, padding='same')(c) # 2-о1 параметр премещается на 2 по вертикали и по горизонтали

    return c, p
    
# up_sampling block for NN
def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    # us = keras.layers.UpSampling2D((2, 2))(x)
    us = keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding=padding)(x) # formula: output_size=(input_size−1)×strides+kernel_size
    us = keras.layers.BatchNormalization()(us)
    us = keras.layers.LeakyReLU()(us)
    concat = keras.layers.Concatenate()([us, skip])

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(concat)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    c = keras.layers.Dropout(0.25)(c)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    # print(f"us shape: {c.shape}")

    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(128, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    c = keras.layers.Dropout(0.25)(c)

    c = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    c = keras.layers.Dropout(0.3)(c)

    c = keras.layers.Conv2D(128, kernel_size=kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    return c 

# Unet architecture
def Unet():

    inputs = keras.layers.Input((image_size, image_size, 2))

    c1, p1 = down_block(inputs, 16) 
    c2, p2 = down_block(p1, 32)
    c3, p3 = down_block(p2, 64)
    # c4, p4 = down_block(p3, 128)

    bn = bottleneck(p3, 256)

    # u1 = up_block(bn, c4, 128)
    u2 = up_block(bn, c3, 64)
    u3 = up_block(u2, c2, 32) 
    u4 = up_block(u3, c1, 16) 

    outputs = keras.layers.Conv2D(1, (3,3), padding="same")(u4)
    # outputs = keras.layers.Conv2D(1, (3,3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01))(u4)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.LeakyReLU()(outputs)

    model = keras.models.Model(inputs, outputs)

    return model

def scheduler(epoch, lr):
    if epoch == 0:
        return lr
    elif (epoch+1) % 5 == 0:
        return lr / 5
    else:
        return lr

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def pearson_r(y_true, y_pred):
    y_true_temp = y_true
    epsilon = 10e-5
    x = y_true_temp
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return K.mean(r)

class FFTVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, sample_index=0):
        super().__init__()
        self.validation_data = validation_data
        self.sample_index = sample_index
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch) % 5 == 0:  # Визаулизировать каждые 5 эпох
            # Получить батч данных
            for x_batch, y_batch in self.validation_data.take(1):
                # Предсказание модели
                y_pred = self.model.predict(x_batch)
                
                # Визуализация
                visualize_fft_results(y_batch, y_pred, self.sample_index)
                break

if __name__ == "__main__":
    random.seed(0)

    # print(tf.config.list_physical_devices()) # вывод устройств

    # unet parameters
    image_size = 32
    Batch_size = 64
    num_images = 60000
    num_of_folders = 1
    # num_of_folders = int(sys.argv[1])
    # print(num_of_folders)

    img_path = './dataset_32/img/*.png'
    zero_corr_holo = './dataset_32/zero_corr_holo/*.png'
    # true_holo_paths = './dataset_classifier/true_holo/*.png'
    true_holo_paths = './dataset_32/holo/*.png'
    # false_holo_paths = './dataset_classifier/false_holo/*.png'
    # hack_holo_paths = './dataset_classifier/false_holo/*.png'
    phase_file = './dataset_32/phase_1.png'
    phase_img_plane_paths = './dataset_classifier/img_512/*.png'

    img_files = sorted(tf.io.gfile.glob(img_path))
    true_holo_files = sorted(tf.io.gfile.glob(true_holo_paths))
    zero_holo_files = sorted(tf.io.gfile.glob(zero_corr_holo))
    phase_img_plane_files = sorted(tf.io.gfile.glob(phase_img_plane_paths))

    indices = list(range(num_images))
    random.shuffle(indices)

    subset_size = 48000
    train_indices = indices[:subset_size]
    test_indices = indices[subset_size:]

    np.savetxt("./dataset_32/train_indices_25cm.txt", train_indices)
    np.savetxt("./dataset_32/test_indices_25cm.txt", test_indices)

    train_ds = create_dataset(img_files, phase_file, [zero_holo_files, true_holo_files, phase_img_plane_files], train_indices, num_of_folders)
    test_ds = create_dataset(img_files, phase_file, [zero_holo_files, true_holo_files, phase_img_plane_files], test_indices, num_of_folders)

    train_ds = train_ds.shuffle(buffer_size=500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.shuffle(buffer_size=500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    base_dir = 'D:/Person/Prokudin/dataset_32'
    checkpoint_path = base_dir + '/checkpoint/model_epoch_{epoch}.h5'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_weights_only=False,
                                                    save_freq='epoch',
                                                    verbose=1)
    lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    terminate_on_NaN = keras.callbacks.TerminateOnNaN()
    alpha_scheduler = AlphaScheduler(initial_alpha, final_alpha, total_epochs)
    alpha_scheduler2 = AlphaScheduler2(5, current_alpha)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  
    fft_callback = FFTVisualizationCallback(test_ds, sample_index=0)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0, verbose=1)

    model = Unet()
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001), loss='mse', metrics=[]) 
    # model.summary()

    # training the model
    model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=[terminate_on_NaN, checkpoint_callback])

    # # Save the Weights
    # model.save_weights(f"./dataset_64x2_cifar10/60kfalsephases.weights.h5")

    # # Save the Model
    # model.save(f"./dataset_64x2_cifar10/60kflasephases.h5")