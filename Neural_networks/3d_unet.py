import numpy as np
import tensorflow as tf
import keras
import pathlib
import matplotlib.pyplot as plt

# unet parameters
image_size = 128
epochs = 5
BATCH_SIZE = 5

# loading img
def load_image(image_file):
    image = tf.io.read_file(image_file) # This operation returns a tensor with the entire contents of the input filename
    image = tf.image.decode_png(image, channels=1, dtype=tf.dtypes.uint8) # 3d variable doesn't make a difference 
    image = tf.cast(image, tf.float32) / 255 # Normalization

    return image

# loading img & img-label
def load_image_pair(dir_img_1, dir_img_2, dir_holo):
    img_input = tf.concat([load_image(dir_img_1), load_image(dir_img_2)], axis=2)
    return img_input, load_image(dir_holo)

# improving performance
def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds

img_path = 'C:/Users/Proku/OneDrive/Desktop/kinoform/dataset_128/img/*.png'
holo_path = 'C:/Users/Proku/OneDrive/Desktop/kinoform/dataset_128/holo/*.png'

ds1 = tf.data.Dataset.list_files(img_path, shuffle=False) # (str(pathlib.Path('C:/Users/Proku/OneDrive/Desktop/kinoform/dataset_512')/'img/*'), shuffle=False)
ds2 = tf.data.Dataset.list_files(holo_path, shuffle=False)

subset_size = 10000
train_img_1 = ds1.take(subset_size)
train_img_2 = ds1.skip(subset_size).take(subset_size)
test_img_1 = ds1.skip(20000).take(1000)
test_img_2 = ds1.skip(21000).take(1000)
train_holo = ds2.take(subset_size)
test_holo = ds2.skip(subset_size).take(1000)

train_img = tf.data.Dataset.zip((train_img_1, train_img_2, train_holo))
test_img = tf.data.Dataset.zip((test_img_1, test_img_2, test_holo))

print(tf.data.experimental.cardinality(train_img).numpy())

# Преобразование списка путей к папкам в пары изображений и оптимизация производительности
train_img = train_img.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
train_img = configure_for_performance(train_img)
test_img = test_img.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
test_img = configure_for_performance(test_img)



# получаем батч для проверки изображений в данных для нейросети
# image_batch, _ = next(iter(train_img)) # next -> returns only one batch it's good for checking the data
# plt.imshow(image_batch[0])
# plt.show()

# down_sampling block for NN
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.ReLU()(c)

    # c = keras.layers.Dropout(0.1)(c)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.ReLU()(c)

    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c) # 2-о1 параметр премещается на 2 по вертикали и по горизонтали

    return c, p

# first down_sampling block for NN
def down_block1(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    # c = keras.layers.Conv2D(filters, (5, 5), padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, (5, 5), padding=padding, strides=strides)(x)
    c = keras.layers.BatchNormalization()(c)    
    c = keras.layers.ReLU()(c)

    # c = keras.layers.Dropout(0.1)(c)

    c = keras.layers.Conv2D(filters, (5, 5), padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)    
    c = keras.layers.ReLU()(c)

    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c) # 2-о1 параметр премещается на 2 по вертикали и по горизонтали

    return c, p

# up_sampling block for NN
def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    # us = keras.layers.UpSampling2D((2, 2))(x)
    us = keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding=padding)(x) # formula: output_size=(input_size−1)×strides+kernel_size
    us = keras.layers.ReLU()(us)
    us = keras.layers.BatchNormalization()(us)
    concat = keras.layers.Concatenate()([us, skip])

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(concat)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.ReLU()(c)

    # c = keras.layers.Dropout(0.2)(c)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.ReLU()(c)

    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.ReLU()(c)

    # c = keras.layers.Dropout(0.3)(c)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.ReLU()(c)

    return c 

# Unet architecture
def Unet():
    # f = [4, 8, 16, 32, 64, 256, 256]
    inputs = keras.layers.Input((image_size, image_size, 2))
    
    p0 = inputs
    # c1, p1 = down_block1(p0, f[0]) 
    c2, p2 = down_block(p0, 4) 
    c3, p3 = down_block(p2, 12) 
    c4, p4 = down_block(p3, 36) 
    c5, p5 = down_block(p4, 72)
    
    bn = bottleneck(p5, 512)

    u1 = up_block(bn, c5, 256) 
    u2 = up_block(u1, c4, 128) 
    u3 = up_block(u2, c3, 64) 
    u4 = up_block(u3, c2, 32)
    # u5 = up_block(u4, c1, f[0]) 

    outputs = keras.layers.Conv2D(1, (3,3), padding="same", activation="linear")(u4)
    model = keras.models.Model(inputs, outputs)

    return model

lr = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.002, decay_steps=2, decay_rate=0.8)

plateu = keras.callbacks.ReduceLROnPlateau()

def scheduler(epoch, lr):
    if epoch == 0:
        return lr
    elif (epoch % 2) == 0:
        return lr / 2
    else:
        return lr
    
callbck = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

model = Unet()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss = 'mean_squared_error') # lambda x,y: custom_loss(x,y)) # x - true, y - predictions
model.summary()

# training the model
model.fit(train_img, validation_data=test_img, epochs=epochs, callbacks=callbck) #, batch_size=BATCH_SIZE)


# Save the Weights
model.save_weights("./UNet3D.weights.h5")

# Save the Model
model.save("./UNet3D.h5")
