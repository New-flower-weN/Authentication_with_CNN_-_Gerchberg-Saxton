import sys
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.append('..')
from src.data_loader import DataLoader
from config.config import BATCH_SIZE

def main():
    model = tf.keras.models.load_model('./data/dataset_32/model_final.h5')
    
    data_loader = DataLoader(
        img_path='./data/dataset_32/img/*.png',
        true_holo_path='./data/dataset_32/holo/*.png',
        zero_holo_path='./data/dataset_32/zero_corr_holo/*.png',
        phase_path='./data/dataset_32/phase_1.png',
        phase_img_plane_path='./data/dataset_32/img/*.png'
    )
    
    _, test_ds, _, _ = data_loader.get_datasets(BATCH_SIZE)
    
    loss = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {loss}")
    
    sample_batch = next(iter(test_ds))
    predictions = model.predict(sample_batch[0])
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        axes[0, i].imshow(sample_batch[1][i].numpy().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Ground Truth {i}')
        axes[1, i].imshow(predictions[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Prediction {i}')
    
    plt.tight_layout()
    plt.savefig('./results/predictions.png')
    plt.show()

if __name__ == "__main__":
    main()