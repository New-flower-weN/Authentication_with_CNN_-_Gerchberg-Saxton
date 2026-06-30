import numpy as np
import cv2
from tqdm import tqdm
from config.config import N, ps, lmbd, z

class ZeroCorrelationGenerator:
    """Generator for zero correlation holograms"""
    
    def __init__(self, N=N, ps=ps, lmbd=lmbd, z=z):
        self.N = N
        self.ps = ps
        self.lmbd = lmbd
        self.z = z
        
        # Setup grid
        X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
        self.du = lmbd * z / (N * ps)
        u = ps * X
        v = ps * Y
        self.wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)
        
        # Load reference phase
        self.phase = cv2.imread('./data/dataset_32/phase_1.png', cv2.IMREAD_GRAYSCALE)
        self.phase = self.phase - np.mean(self.phase)
        self.phase = self.phase / np.std(self.phase)
        self.phase = self.phase.flatten()
        
    def orthogonalize(self, holo):
        """Remove correlation with reference phase"""
        holo = holo - np.mean(holo)
        holo = holo / np.std(holo)
        holo = holo.flatten()
        holo = holo - np.dot(holo, self.phase) / np.dot(self.phase, self.phase) * self.phase
        holo = holo.reshape((self.N, self.N))
        return holo
    
    def generate_dataset(self, start_idx, end_idx, input_dir, output_dir):
        """Generate zero correlation holograms for dataset"""
        for i in tqdm(range(start_idx, end_idx)):
            holo = cv2.imread(f'{input_dir}/holo/{i}.png', cv2.IMREAD_GRAYSCALE) / 255.0
            
            # Orthogonalize
            holo = self.orthogonalize(holo)
            
            # Normalize
            holo = (holo - np.min(holo)) / (np.max(holo) - np.min(holo))
            holo = np.uint8(holo * 255)
            
            cv2.imwrite(f'{output_dir}/zero_corr_holo/{i}.png', holo)