import  numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import IPython.display as display
#from PIL import Image
import tensorflow as tf
from keras import datasets, layers, models, losses
#import tensorflow_datasets as tfds
from keras import backend as K
#from keras.callbacks import TensorBoard
import glob
import time
import functools
import random
import pathlib
import os
import datetime
from datetime import datetime, date, time
import cv2
import keras
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

NUM_EPOCHS = 10

def load_image1(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1, dtype=tf.dtypes.uint8)
    image = tf.image.resize(image, [128, 128])
    #image = cv2.resize(image, (80, 80))
    image /= 255
    #image = image[tf.newaxis, :]
    return image


def load_image2(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1, dtype=tf.dtypes.uint8)
    # image = tf.image.resize(image, [128, 128])
    #image = cv2.resize(image, (80, 80))
    image /= 255
    #image = image[tf.newaxis, :]
    return image

def load_dataset(int_path, phase_path):
    return load_image1(int_path), load_image2(phase_path)


def process_image(image):
    image = image[0, :, :, 0]
    image -= np.amin(image)
    image = (image / np.amax(image)) * 65535
    return image


def downsampling_block_y(inputs, filter_count):

    initializer = keras.initializers.glorot_uniform()
    regularizer = None
    #regularizer = tf.keras.regularizers.l1(0.01)

    x_input = inputs

    x_1 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(inputs)
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = keras.activations.swish(x_1)
    x_1 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x_1)
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = keras.activations.swish(x_1)
    x_1 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x_1)
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = keras.activations.swish(x_1)
    x_1 = layers.Add()([x_1, x_input])

    x_2 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(inputs)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = keras.activations.swish(x_2)
    x_2 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x_2)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = tf.keras.activations.swish(x_2)
    x_2 = layers.Add()([x_2, x_input])

    x_3 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(inputs)
    x_3 = layers.BatchNormalization()(x_3)
    x_3 = tf.keras.activations.swish(x_3)
    x_3 = layers.Add()([x_3, x_input])

    x = layers.Concatenate()([x_1, x_2, x_3])
    x_skip = x
    x = layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(x)

    return x, x_skip


def downsampling_block_1(inputs, filter_count):

    initializer = tf.keras.initializers.glorot_uniform()
    regularizer = None
    #regularizer = tf.keras.regularizers.l1(0.01)

    x_input = inputs

    x_2 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(inputs)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = layers.LeakyReLU()(x_2)
    x_2 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x_2)
    x_2 = layers.BatchNormalization()(x_2)
    x = layers.LeakyReLU()(x_2)

    x_skip = x
    x = layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(x)

    return x, x_skip


def downsampling_block_2(inputs, filter_count):

    initializer = keras.initializers.glorot_uniform()
    regularizer = None
    # regularizer = tf.keras.regularizers.l1(0.01)

    x_input = inputs

    x_1 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer,
                        padding='same')(inputs)
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = layers.LeakyReLU()(x_1)
    x_2 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer,
                        padding='same')(inputs)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = layers.LeakyReLU()(x_2)
    x_2 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer,
                        padding='same')(x_2)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = layers.LeakyReLU()(x_2)
    x = layers.Add()([x_1, x_2])

    x_skip = x
    x = layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(x)

    return x, x_skip


def downsampling_block_3(inputs, filter_count):

    initializer = tf.keras.initializers.glorot_uniform()
    regularizer = None
    # regularizer = tf.keras.regularizers.l1(0.01)

    x_input = inputs

    x_2 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer,
                        padding='same')(inputs)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = tf.keras.activations.swish(x_2)
    x_2 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer,
                        padding='same')(x_2)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = tf.keras.activations.swish(x_2)
    x = layers.Concatenate()([x_2, x_input])
    x_2 = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer,
                        padding='same')(x_2)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = tf.keras.activations.swish(x_2)

    x_skip = x
    x = layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(x)

    return x, x_skip


def upsampling_block_y(inputs, skip_inputs, filter_count):

    initializer = tf.keras.initializers.glorot_uniform()
    regularizer = None
    # regularizer = tf.keras.regularizers.l1(0.01)

    filters = inputs.shape[3]

    x = layers.Conv2DTranspose(filters, 3, strides=2, kernel_initializer=initializer, kernel_regularizer=regularizer,  padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.swish(x)
    x = layers.Concatenate()([x, skip_inputs])
    x = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.activations.swish(x)

    if filter_count == 256:
        x = layers.Conv2D(128, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = tf.keras.activations.swish(x)
    elif filter_count == 128:
        x = layers.Conv2D(64, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = tf.keras.activations.swish(x)
    elif filter_count == 64:
        x = layers.Conv2D(32, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = tf.keras.activations.swish(x)
    elif filter_count == 32:
        x = layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = keras.activations.swish(x)

    return x


def upsampling_block_1(inputs, skip_inputs, filter_count):

    initializer = tf.keras.initializers.glorot_uniform()
    regularizer = None
    # regularizer = tf.keras.regularizers.l1(0.01)

    filters = inputs.shape[3]

    x = layers.Conv2DTranspose(filter_count, 2, strides=2, kernel_initializer=initializer, kernel_regularizer=regularizer,  padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Concatenate()([x, skip_inputs])
    x = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(filter_count, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer,
                      padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    return x


def upsampling_block_2(inputs, skip_inputs, filter_count):

    initializer = keras.initializers.glorot_uniform()
    regularizer = None
    # regularizer = tf.keras.regularizers.l1(0.01)

    filters = inputs.shape[3]

    x = layers.Conv2DTranspose(64, 2, strides=2, kernel_initializer=initializer, kernel_regularizer=regularizer,  padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Concatenate()([x, skip_inputs])
    x = layers.Conv2D(64, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer,
                      padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(1, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer,
                      padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    return x


def mid_block(inputs):

    initializer = tf.keras.initializers.glorot_uniform()
    regularizer = None
    # regularizer = tf.keras.regularizers.l1(0.01)
    filters = inputs.shape[3]

    x = layers.Conv2D(512, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(1024, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(512, 3, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizer, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    return x



def make_model():

    inputs = tf.keras.layers.Input(shape=[128, 128, 1])

    # initializer = tf.keras.initializers.glorot_uniform()
    # regularizer = None
    #regularizer = tf.keras.regularizers.l1(0.01)

    x, x_skip160 = downsampling_block_2(inputs, 8)
    x, x_skip80 = downsampling_block_2(x, 24)
    x, x_skip40 = downsampling_block_2(x, 72)
    x, x_skip20 = downsampling_block_2(x, 216)

    x = mid_block(x)

    x = upsampling_block_1(x, x_skip20, 256)
    x = upsampling_block_1(x, x_skip40, 128)
    x = upsampling_block_1(x, x_skip80, 64)
    x = upsampling_block_2(x, x_skip160, 32)


    return keras.Model(inputs=inputs, outputs=x)

data_path_input = 'G:/models/REC/data/L_64_50/*.png'
data_path_target = 'G:/models/REC/data/H_k_64_50_3/K/*.png'
data_path_input_val = 'G:/models/REC/data/L_64_val/*.png'
data_path_target_val = 'G:/models/REC/data/H_k_64_50_3_val/*.png'

BATCH_SIZE = 50
BUFFER_SIZE = 10000
AUTOTUNE = tf.data.experimental.AUTOTUNE
CPU_THREADS = 12

ds1 = tf.data.Dataset.list_files(data_path_input, shuffle=False)
ds2 = tf.data.Dataset.list_files(data_path_target, shuffle=False)
ds = tf.data.Dataset.zip((ds1, ds2))
ds = ds.map(load_dataset, num_parallel_calls=CPU_THREADS)
ds = ds.shuffle(buffer_size=BUFFER_SIZE)
ds = ds.repeat(1)
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(AUTOTUNE)

ds_val_1 = tf.data.Dataset.list_files(data_path_input_val, shuffle=False)
ds_val_2 = tf.data.Dataset.list_files(data_path_target_val, shuffle=False)
ds_val = tf.data.Dataset.zip((ds_val_1, ds_val_2))
ds_val = ds_val.map(load_dataset, num_parallel_calls=CPU_THREADS)
ds_val = ds_val.shuffle(buffer_size=BUFFER_SIZE)
ds_val = ds_val.repeat(1)
ds_val = ds_val.batch(BATCH_SIZE)
ds_val = ds_val.prefetch(AUTOTUNE)

model = make_model()



model.compile(optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
              loss=tf.keras.losses.mean_squared_error)


base_dir = 'G:/models/REC/'
data_path = base_dir + 'data/'
checkpoint_path = base_dir + 'checkpoints/K_GS_q1_2024/H_q1_q_4_1_cp-{epoch:04d}.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)

plateau = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                               factor=0.1,
                                               patience=5,
                                               verbose=1,
                                               min_delta=0.0001,
                                               cooldown=3,
                                               min_lr=0.000002)


def scheduler(epoch, lr):
    if epoch == 0:
        return lr
    elif (epoch % 5) == 0:
        return lr / 2
    elif (epoch % 5) != 0:
        return lr


lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
terminate_on_NaN = keras.callbacks.TerminateOnNaN()

model.fit(ds, epochs=NUM_EPOCHS, validation_data=ds_val, callbacks=[terminate_on_NaN, lr_scheduler, checkpoint_callback])
#model.summary()
