import tensorflow as tf
import cv2

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

print(len(train_images), test_images.shape)

for num, img in enumerate(train_images):
    save_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #print(save_img.dtype)
    cv2.imwrite(f'./dataset_32/img/{100001+num}.png', save_img)

for num, img in enumerate(test_images):
    save_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #print(save_img.dtype)
    cv2.imwrite(f'./dataset_32/img/{150001+num}.png', save_img)
