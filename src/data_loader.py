import tensorflow as tf
import random
from config.config import N

def load_and_process_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1, dtype=tf.dtypes.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def create_dataset(img_files, phase_file, holo_files, indices):
    img_dataset = []
    holo_dataset_true = []
    holo_dataset_zero_corr = []
    phase_dataset = []

    for idx in indices:
        img_dataset.append(img_files[idx])
        phase_dataset.append(phase_file)
        holo_dataset_zero_corr.append(holo_files[0][idx])
        holo_dataset_true.append(holo_files[1][idx])

    ds = tf.data.Dataset.from_tensor_slices(
        ((img_dataset, phase_dataset), 
         (holo_dataset_true, holo_dataset_zero_corr))
    )
    
    def process_nested_files(file_paths, holo_path):
        img_path, phase_path = file_paths
        holo_true_path, holo_zero_path = holo_path
        
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

class DataLoader:
    def __init__(self, img_path, true_holo_path, zero_holo_path, 
                 phase_path, num_images=60000):
        self.img_files = sorted(tf.io.gfile.glob(img_path))
        self.true_holo_files = sorted(tf.io.gfile.glob(true_holo_path))
        self.zero_holo_files = sorted(tf.io.gfile.glob(zero_holo_path))
        self.phase_file = phase_path
        self.num_images = num_images
        
    def get_train_test_split(self, subset_size=48000):
        indices = list(range(self.num_images))
        random.shuffle(indices)
        
        train_indices = indices[:subset_size]
        test_indices = indices[subset_size:]
        
        return train_indices, test_indices
    
    def get_datasets(self, batch_size=64):
        train_indices, test_indices = self.get_train_test_split()
        num_of_folders = 1
        
        holo_files = [self.zero_holo_files, self.true_holo_files, 
                      self.phase_img_plane_files]
        
        train_ds = create_dataset(self.img_files, self.phase_file, 
                                  holo_files, train_indices, num_of_folders)
        test_ds = create_dataset(self.img_files, self.phase_file, 
                                 holo_files, test_indices, num_of_folders)
        
        train_ds = (train_ds
                   .shuffle(buffer_size=500)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))
        test_ds = (test_ds
                  .shuffle(buffer_size=500)
                  .batch(batch_size)
                  .prefetch(tf.data.AUTOTUNE))
        
        return train_ds, test_ds, train_indices, test_indices