import numpy as np

# Physical parameters
N = 32
ps = 7.8e-6  # pixel size (m)
lmbd = 561.0e-9  # wavelength (m)
z = 0.2  # propagation distance (m)
z2 = 0.4  # second propagation distance (m)

# Dataset parameters
NUM_IMAGES = 60000
BATCH_SIZE = 64
EPOCHS = 100
PHASE_GRAD = 16

# Paths
BASE_PATH = './data/dataset_32'
IMG_PATH = f'{BASE_PATH}/img/*.png'
HOLO_PATH = f'{BASE_PATH}/holo/*.png'
ZERO_CORR_PATH = f'{BASE_PATH}/zero_corr_holo/*.png'
PHASE_PATH = f'{BASE_PATH}/phase_1.png'

# Model parameters
IMAGE_SIZE = 32
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 1