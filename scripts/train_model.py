import sys
import os
import tensorflow as tf
import keras
sys.path.append('..')
from src.data_loader import DataLoader
from src.model import Unet
from src.utils import save_training_indices
from config.config import BATCH_SIZE, EPOCHS, IMAGE_SIZE

def main():
    base_dir = './data/dataset_32'
    checkpoint_path = f'{base_dir}/checkpoint/model_epoch_{{epoch}}.h5'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=False, save_freq='epoch', verbose=1
    )
    terminate_on_NaN = keras.callbacks.TerminateOnNaN()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=2, min_lr=0.0, verbose=1
    )
    
    data_loader = DataLoader(
        img_path='./data/dataset_32/img/*.png',
        true_holo_path='./data/dataset_32/holo/*.png',
        zero_holo_path='./data/dataset_32/zero_corr_holo/*.png',
        phase_path='./data/dataset_32/phase_1.png',
        phase_img_plane_path='./data/dataset_32/img/*.png'
    )
    
    train_ds, test_ds, train_indices, test_indices = data_loader.get_datasets(BATCH_SIZE)
    save_training_indices(train_indices, test_indices)
    
    model = Unet(image_size=IMAGE_SIZE)
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001), loss='mse', metrics=[])
    
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=[terminate_on_NaN, checkpoint_callback, early_stopping, reduce_lr]
    )
    
    model.save(f'{base_dir}/model_final.h5')

if __name__ == "__main__":
    main()