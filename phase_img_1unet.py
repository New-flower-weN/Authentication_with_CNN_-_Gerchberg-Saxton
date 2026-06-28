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
z = 0.2 
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
        phase = load_and_process_image(phase_path)
        holo = load_and_process_image(holo_true_path)
        zero_holo = load_and_process_image(holo_zero_path)


        phase = tf.reshape(phase, (N, N, 1))
        holo = tf.reshape(holo, (N, N, 1))
        zero_holo = tf.reshape(zero_holo, (N, N, 1))
        input_tensor = tf.concat([zero_holo, phase], axis=-1)

        return input_tensor, holo
    
    ds = ds.map(process_nested_files, num_parallel_calls=tf.data.AUTOTUNE)

    return ds

# Блоки для ветки энкодера 
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
    
# Блоки для ветки декодера
def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
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
    terminate_on_NaN = keras.callbacks.TerminateOnNaN()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  
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