import tensorflow as tf
import keras
from keras import backend as K

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)
    c = keras.layers.Dropout(0.2)(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.LeakyReLU()(c)
    p = keras.layers.MaxPool2D(pool_size=2, strides=None, padding='same')(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding=padding)(x)
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

def Unet(image_size=32):
    inputs = keras.layers.Input((image_size, image_size, 2))
    
    # encoder
    c1, p1 = down_block(inputs, 16)
    c2, p2 = down_block(p1, 32)
    c3, p3 = down_block(p2, 64)
    
    # bottleneck
    bn = bottleneck(p3, 256)
    
    # decoder
    u2 = up_block(bn, c3, 64)
    u3 = up_block(u2, c2, 32)
    u4 = up_block(u3, c1, 16)
    
    outputs = keras.layers.Conv2D(1, (3, 3), padding="same")(u4)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.LeakyReLU()(outputs)
    
    model = keras.models.Model(inputs, outputs)
    return model