import numpy as np
import cv2
from tqdm import tqdm
import cupy as cp
from cupy.fft import fft2, ifft2, fftshift, ifftshift
from config.config import N, ps, lmbd, z, PHASE_GRAD

class HologramGenerator:
    def __init__(self, N=N, ps=ps, lmbd=lmbd, z=z, phase_grad=PHASE_GRAD):
        self.N = N
        self.ps = ps
        self.lmbd = lmbd
        self.z = z
        self.phase_grad = phase_grad
        
        self.initial_phase = np.random.randint(0, 16, (N, N))
        self.initial_phase = (self.initial_phase / 16) * 2 * np.pi - np.pi
        self.initial_phase = np.repeat(self.initial_phase.reshape((1, N, N)), 50, axis=0)
        
        X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
        self.du = lmbd * z / (N * ps)
        u = ps * X
        v = ps * Y
        self.wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)
        self.cp_wave_front = cp.asarray(self.wave_front)
        
    def generate_single(self, img, height, width, iterations=100):
        img_amp = 1 - np.sqrt(img)
        phase = self.initial_phase
        
        for i in range(iterations):
            complex_amp = 1 * np.exp(1j * phase)
            freq_img = fftshift(fft2(ifftshift(complex_amp * np.exp(self.wave_front))))
            phase1 = np.angle(freq_img)
            freq_new = img_amp * np.exp(1j * phase1)
            img_plane = ifftshift(ifft2(fftshift(freq_new))) * np.exp(-self.wave_front)
            phase = np.angle(img_plane)
        
        return np.uint8((phase) * 255 / (2 * np.pi))
    
    def generate_batch(self, img_arr, iterations=100):
        img_amp = cp.sqrt(img_arr)
        phase = self.initial_phase
        
        for i in range(iterations):
            complex_amp = 1 * cp.exp(1j * cp.asanyarray(phase))
            freq_img = fftshift(fft2(ifftshift(complex_amp * cp.exp(self.cp_wave_front), 
                                                axes=(1,2)), axes=(1,2)), axes=(1,2))
            phase = cp.angle(freq_img)
            freq_new = img_amp * np.exp(1j * phase)
            img_plane = ifftshift(ifft2(fftshift(freq_new, axes=(1,2)), axes=(1,2)), 
                                  axes=(1,2)) * cp.exp(-self.cp_wave_front)
            phase = cp.angle(img_plane) + cp.pi
            
            phase = (phase + cp.pi) / (2 * cp.pi)
            phase = ((phase - 1/(2*self.phase_grad)) * self.phase_grad)
            phase = (phase.round() * 2 * cp.pi / self.phase_grad) - cp.pi
        
        phase_max = cp.max(phase, axis=(1,2), keepdims=True)
        phase = np.uint8((phase.get() / phase_max.get()) * 255)
        return phase
    
    def generate_dataset(self, start_idx, end_idx, output_dir):
        img_arr = cp.empty((50, self.N, self.N))
        j = 0
        
        for i in tqdm(range(start_idx, end_idx)):
            img_arr[j] = cp.asarray(cv2.imread(f'{output_dir}/img/{i}.png', 
                                               cv2.IMREAD_GRAYSCALE))
            
            if j == 49:
                holo_arr = self.generate_batch(img_arr)
                for k in range(i-49, i+1):
                    cv2.imwrite(f'{output_dir}/holo/{k}.png', holo_arr[49-j])
                    j -= 1
            j += 1
        
    def save_initial_phase(self, output_path):
        phase_img = np.uint8((self.initial_phase[0] + np.pi) / (2 * np.pi) * 255)
        cv2.imwrite(output_path, phase_img)