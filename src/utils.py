import numpy as np
from skimage.metrics import structural_similarity as ssim
from ..config.config import BASE_PATH

def NRMSE(x, y): # normalized root mean square error
    mse = np.sum(x * y)**2
    sum_sq = np.sum(x**2) * np.sum(y**2)
    nrmse = np.sqrt(1 - mse / sum_sq)
    return nrmse

def save_training_indices(train_indices, test_indices, base_dir=f'{BASE_PATH}'):
    np.savetxt(f"{base_dir}/train_indices.txt", train_indices)
    np.savetxt(f"{base_dir}/test_indices.txt", test_indices)

def load_training_indices(base_dir=f'{BASE_PATH}'):
    train_indices = np.loadtxt(f"{base_dir}/train_indices.txt").astype(np.int64)
    test_indices = np.loadtxt(f"{base_dir}/test_indices.txt").astype(np.int64)
    return train_indices, test_indices