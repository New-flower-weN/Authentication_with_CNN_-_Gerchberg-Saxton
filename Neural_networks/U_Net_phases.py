import tensorflow as tf
import numpy as np
import keras
import random
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# setting physical parameters
N = 64
N1 = N - 1
ps = 8.0e-6
lmbd = 633.0e-9
BATCH_SIZE = 50
epochs = 25
z = np.sqrt(N * N) * ps**2 / lmbd 
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

def visualize_fft_results(y_true, y_pred, sample_index=0):
    """
    Визуализация FFT результатов для одного примера из батча
    """
    # Убираем последнюю ось (каналы), если она равна 1
    y_true_2d = tf.squeeze(y_true)
    y_pred_2d = tf.squeeze(y_pred)

    y_true_2d = tf.cast(y_true_2d, tf.complex64)
    y_pred_2d = tf.cast(y_pred_2d, tf.complex64)
    
    # Compute FFTs
    y_true_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*y_true_2d*2*np.pi-np.pi) * np.exp(wave_front), axes=(1, 2))), axes=(1, 2)) 
    y_pred_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*y_pred_2d*2*np.pi-np.pi) * np.exp(wave_front), axes=(1, 2))), axes=(1, 2)) 

    # Calculate magnitudes
    y_true_fft_abs = tf.abs(y_true_fft).numpy()[sample_index, :, :]
    y_pred_fft_abs = tf.abs(y_pred_fft).numpy()[sample_index, :, :]
    
    # Normalize
    max_val_true = np.max(y_true_fft_abs)
    y_true_fft_abs_normalized = y_true_fft_abs / max_val_true
    
    max_val_pred = np.max(y_pred_fft_abs)
    y_pred_fft_abs_normalized = y_pred_fft_abs / max_val_pred
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(y_true[sample_index, :, :, 0], cmap='gray')
    axes[0, 0].set_title('Original True')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(y_pred[sample_index, :, :, 0], cmap='gray')
    axes[0, 1].set_title('Original Pred')
    axes[0, 1].axis('off')
    
    # FFT magnitudes
    axes[1, 0].imshow(y_true_fft_abs_normalized**2, cmap='gray')
    axes[1, 0].set_title('True FFT Magnitude')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(y_pred_fft_abs_normalized**2, cmap='gray')
    axes[1, 1].set_title('Pred FFT Magnitude')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return y_true_fft_abs_normalized, y_pred_fft_abs_normalized

def custom_loss(y_true, y_pred, alpha=0.01):
    # Убираем последнюю ось (каналы), если она равна 1
    y_true_2d = tf.squeeze(y_true)  # форма: (batch_size, 64, 64)
    y_pred_2d = tf.squeeze(y_pred)  # форма: (batch_size, 64, 64)

    # Cast to complex64 for FFT operations
    y_true_complex = tf.cast(y_true_2d, tf.complex64)
    y_pred_complex = tf.cast(y_pred_2d, tf.complex64)
    
    # Compute FFTs
    y_true_fft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.exp(1j*y_true_complex*2*np.pi-np.pi) * np.exp(wave_front), axes=(1, 2))), axes=(1, 2))
    y_pred_fft = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.fftshift(tf.exp(1j*y_pred_complex*2*np.pi-np.pi) * np.exp(wave_front), axes=(1, 2))), axes=(1, 2))
    
    # Calculate spatial domain loss (using original float tensors)
    spatial_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Calculate frequency domain loss (take magnitude of complex differences)
    y_true_fft_abs = tf.abs(y_true_fft)
    y_pred_fft_abs = tf.abs(y_pred_fft)
    max_val_true = tf.reduce_max(y_true_fft_abs, axis=0)
    y_true_fft_abs = y_true_fft_abs / max_val_true
    max_val_pred = tf.reduce_max(y_pred_fft_abs, axis=0)
    y_pred_fft_abs = y_pred_fft_abs / max_val_pred
    freq_loss = tf.reduce_mean(tf.square(y_true_fft_abs - y_pred_fft_abs)) 

    return (1 - alpha) * spatial_loss + alpha * freq_loss

def load_and_process_image(file_path):
    """
    Load and process a single image file
    num_of_channels parameter is kept for compatibility but not used in indexing
    """
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1, dtype=tf.dtypes.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def create_dataset(img_files, phase_files, holo_files, indices, num_of_folders):
    img_dataset = []
    holo_dataset = []
    phase_dataset = []

    for idx in indices:
        img_dataset.append(img_files[idx])
        folder = random.randint(0, num_of_folders-1)
        phase_dataset.append(phase_files[folder][idx])
        holo_dataset.append(holo_files[folder][idx])

    # Create dataset with tuple structure
    ds = tf.data.Dataset.from_tensor_slices(((img_dataset, phase_dataset), holo_dataset))
    
    # Map function that handles the nested structure
    def process_nested_files(file_paths, holo_path):
        img_path, phase_path = file_paths
        img = load_and_process_image(img_path)
        phase = load_and_process_image(phase_path)
        holo = load_and_process_image(holo_path)

        img = tf.reshape(img, (64, 64, 1))
        phase = tf.reshape(phase, (64, 64, 1))

        input_tensor = tf.concat([img, phase], axis=-1)

        return input_tensor, holo
    
    ds = ds.map(process_nested_files, num_parallel_calls=tf.data.AUTOTUNE)

    return ds

# down_sampling block for NN
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    # c = keras.layers.Dropout(0.1)(c)

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

    # c = keras.layers.Dropout(0.2)(c)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    # print(f"us shape: {c.shape}")

    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(512, kernel_size=kernel_size, padding=padding, strides=strides)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    # c = keras.layers.Dropout(0.2)(c)

    c = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    # c = keras.layers.Dropout(0.2)(c)

    c = keras.layers.Conv2D(512, kernel_size=kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)

    return c 

# Unet architecture
def Unet():

    inputs = keras.layers.Input((image_size, image_size, 2))

    c3, p3 = down_block(inputs, 64) 
    c4, p4 = down_block(p3, 128)
    c5, p5 = down_block(p4, 256)

    bn = bottleneck(p5, 1024)

    u1 = up_block(bn, c5, 256)
    u2 = up_block(u1, c4, 128)
    u3 = up_block(u2, c3, 64) 

    outputs = keras.layers.Conv2D(1, (3,3), padding="same")(u3)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.LeakyReLU()(outputs)

    model = keras.models.Model(inputs, outputs)

    return model

def scheduler(epoch, lr):
    if epoch == 0:
        return lr
    elif (epoch % 5 == 0):
        return lr / 2
    else:
        return lr

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

class FFTVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, sample_index=0):
        super().__init__()
        self.validation_data = validation_data
        self.sample_index = sample_index
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Визаулизировать каждые 5 эпох
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
    image_size = 64
    Batch_size = 50
    num_images = 60000
    num_of_folders = 1
    # num_of_folders = int(sys.argv[1])
    # print(num_of_folders)

    img_path = './dataset_64_cifar10/img/*.png'
    holo_paths = ['./dataset_64x2_cifar10/GS_dataset_main_phase/*.png']
    phase_paths = ['./dataset_64x2_cifar10/phases_all_random/*.png']

    img_files = []
    holo_files = []
    phase_files= []

    img_files = sorted(tf.io.gfile.glob(img_path))
    for i in range(num_of_folders):
        holo_files.append(sorted(tf.io.gfile.glob(holo_paths[i])))
        phase_files.append(sorted(tf.io.gfile.glob(phase_paths[i])))

    indices = list(range(num_images))
    random.shuffle(indices)

    subset_size = 50000
    train_indices = indices[:subset_size]
    test_indices = indices[subset_size:]

    train_ds = create_dataset(img_files, phase_files, holo_files, train_indices, num_of_folders)
    test_ds = create_dataset(img_files, phase_files, holo_files, test_indices, num_of_folders)

    train_ds = train_ds.shuffle(buffer_size=3000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.shuffle(buffer_size=3000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    fft_callback = FFTVisualizationCallback(test_ds, sample_index=0)

    model = Unet()
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.01), loss=custom_loss, metrics=[ssim_metric, 'mse']) 
    # model.summary()

    # training the model
    model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=[lr_scheduler])

    # Save the Weights
    model.save_weights(f"./dataset_64x2_cifar10/60kfalsephases.weights.h5")

    # Save the Model
    model.save(f"./dataset_64x2_cifar10/60kflasephases.h5")