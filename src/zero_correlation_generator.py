import numpy as np
import cv2
from tqdm import tqdm
from config.config import N, ps, lmbd, z, PHASE_PATH, BASE_PATH

class ZeroCorrelationGenerator:
    def __init__(self, N=N, ps=ps, lmbd=lmbd, z=z):
        self.N = N
        self.ps = ps
        self.lmbd = lmbd
        self.z = z
        
        X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
        self.du = lmbd * z / (N * ps)
        u = ps * X
        v = ps * Y
        self.wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)
        
        self.phase = cv2.imread(f'{PHASE_PATH}', cv2.IMREAD_GRAYSCALE)
        self.phase = self.phase - np.mean(self.phase)
        self.phase = self.phase / np.std(self.phase)
        self.phase = self.phase.flatten()
        
    def orthogonalize(self, holo):
        holo = holo - np.mean(holo)
        holo = holo / np.std(holo)
        holo = holo.flatten()
        holo = holo - np.dot(holo, self.phase) / np.dot(self.phase, self.phase) * self.phase
        holo = holo.reshape((self.N, self.N))
        return holo
    
    def generate_dataset(self, start_idx, end_idx, input_dir, output_dir):
        for i in tqdm(range(start_idx, end_idx)):
            holo = cv2.imread(f'{input_dir}/holo/{i}.png', cv2.IMREAD_GRAYSCALE) / 255.0
            
            holo = self.orthogonalize(holo)
            
            holo = (holo - np.min(holo)) / (np.max(holo) - np.min(holo))
            holo = np.uint8(holo * 255)
            
            cv2.imwrite(f'{output_dir}/zero_corr_holo/{i}.png', holo)