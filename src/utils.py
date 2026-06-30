import numpy as np
from skimage.metrics import structural_similarity as ssim

def NRMSE(x, y):
    """Calculate Normalized Root Mean Square Error"""
    mse = np.sum(x * y)**2
    sum_sq = np.sum(x**2) * np.sum(y**2)
    nrmse = np.sqrt(1 - mse / sum_sq)
    return nrmse

def create_matrix_numpy(rows, cols, zero_fraction=0.5):
    """
    Create matrix with given fraction of zeros
    """
    total_elements = rows * cols
    num_zeros = int(total_elements * zero_fraction)
    matrix = np.ones((rows, cols), dtype=int)
    indices = np.random.choice(total_elements, num_zeros, replace=False)
    row_indices = indices // cols
    col_indices = indices % cols
    matrix[row_indices, col_indices] = 0
    return matrix

def save_training_indices(train_indices, test_indices, base_dir='./data/dataset_32'):
    """Save train/test indices for reproducibility"""
    np.savetxt(f"{base_dir}/train_indices.txt", train_indices)
    np.savetxt(f"{base_dir}/test_indices.txt", test_indices)

def load_training_indices(base_dir='./data/dataset_32'):
    """Load train/test indices"""
    train_indices = np.loadtxt(f"{base_dir}/train_indices.txt").astype(np.int64)
    test_indices = np.loadtxt(f"{base_dir}/test_indices.txt").astype(np.int64)
    return train_indices, test_indices