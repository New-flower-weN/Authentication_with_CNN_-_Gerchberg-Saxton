import numpy as np
# from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import cv2
import cupy as cp  # Если есть GPU NVIDIA
from cupy.fft import fft2, ifft2, fftshift, ifftshift

def cut(arr_img, start_px, end_px):
    return arr_img[start_px:end_px, start_px:end_px]

def NRMSE(x, y):
    mse = np.sum(x * y)**2
    sum_sq = np.sum(x**2) * np.sum(y**2)
    nrmse = np.sqrt(1 - mse / sum_sq)
    return nrmse

# setting physical parameters
PHASE_GRAD = 16
N = 32
N1 = N - 1
ps = 7.8e-6 #8.0e-6 # исходные ключи исп. 7.8e-6
lmbd = 561.0e-9 # 532.0e-9 # исходные ключи исп. 561.0e-9
# z = np.sqrt(N * N) * ps**2 / lmbd # - формулой описывается оптимальное расстояние или типо того; для моих данных z = 0.2 
z = 0.4
z2 = 1.5 * z
du = lmbd * z / (N * ps)
du2 = lmbd * z2 / (N * ps)

initial_phase = np.random.randint(0, 16, (N, N))
initial_phase = (initial_phase / 16 )*2*np.pi - np.pi
cv2.imwrite('./dataset_32/phase_1.png', np.uint8((initial_phase+np.pi)/(2*np.pi)*255))
initial_phase = np.repeat(initial_phase.reshape((1,N,N)), 50, axis=0)

# setting the grid for fourier transformation and implementing the phys. parameters like size of pixel for camera/FTML
X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))

du = lmbd * z / (N * ps)
u = ps * X
v = ps * Y
x = du * X
y = du * Y

# for shortening creating var that'll be under exp()
wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)
wave_front2 = (1j * np.pi / (lmbd * z2)) * (u * u + v * v)

cp_wave_front = cp.asarray((1j * np.pi / (lmbd * z)) * (u * u + v * v))

# size of arrays and number of iterations for the algorithm
iterations = 100
num_of_pics = 10

# arr_it = np.zeros((1000, 2))
jj = 0

# algorithm for 1 picture only, img values should be <=1
def phys_gs_one(img, hight, width):
    img_amp = 1 - np.sqrt(img)
    # phase = np.pi * np.random.uniform(-1, 1, (hight, width)) # np.random.rand(hight, widht)
    # initial_phase = phase
    # phase = phase
    phase = initial_phase
    # arr_ssim = np.zeros(iterations) # metric for compering restored & initial pics

    global jj

    for i in range(iterations):#tqdm(range(iterations)):
        complex_amp = 1 * np.exp(1j * phase)

        freq_img = fftshift(fft2(ifftshift(complex_amp * np.exp(wave_front)))) 

        phase1 = np.angle(freq_img)
        freq_new = img_amp * np.exp(1j * phase1)
        img_plane = ifftshift(ifft2(fftshift(freq_new))) * np.exp(-wave_front) 
        phase = np.angle(img_plane) 

        # I = (np.abs(freq_img)/np.max(np.abs(freq_img)))**2
        # arr_ssim[i] = ssim(img, I, data_range=1)

        # checking = np.corrcoef(((phase + np.pi) / (2 * np.pi)).flatten(), ((initial_phase + np.pi) / (2 * np.pi)).flatten())

        # if checking < 0.1:
            # arr_it[jj,0] = i
            # arr_it[jj,1] = checking
            # jj+=1

            # return np.uint8((phase + np.pi) * 255 / (2*np.pi))

    # plt.title('plotting initial img')
    # plt.imshow(img, cmap='gray')
    # plt.show()

    # plt.title('plotting restored img')
    # plt.imshow(I, cmap='gray')
    # plt.show()

    # plt.title('plotting phase')
    # plt.imshow(phase, cmap='gray')
    # plt.show()

    # plt.title('plotting ssim')
    # plt.plot(arr_ssim)
    # plt.show()

    # phase1 = phase1 + np.pi

    # arr_it[jj,0] = i
    # arr_it[jj,1] = checking
    # jj+=1
    # print(checking)

    return np.uint8((phase) * 255 / (2*np.pi)) #, np.uint8((initial_phase+np.pi) * 255 / (2*np.pi))

# algorith for processing multiple pics at once, img values should be <=1
def phys_gs(img, height=N, width=N):
    # creating amplitude from intencivity from initial pic & creating initial random phase distribution over the grid 
    # (doesn't matter if it's from -pi to pi or from 0 to pi -> algorithm outputs phase ditribution from -pi to pi)
    img_amp = cp.sqrt(img)
    phase =  initial_phase # 100 pics at being processed at once
    # arr_ssim = cp.zeros((num_of_pics, iterations)) # metric for compering restored & initial pics
    arr_corr = cp.zeros((num_of_pics, iterations))
    arr_nstd = cp.zeros((num_of_pics, iterations))

    for i in range(iterations):
        # the initial phase
        complex_amp = 1 * cp.exp(1j * cp.asanyarray(phase))
        # restored img
        freq_img = fftshift(fft2(ifftshift(complex_amp * cp.exp(cp_wave_front), axes=(1,2)), axes=(1,2)), axes=(1,2)) # тут восстановление
        # retrieving the phase
        phase = cp.angle(freq_img)

        # new complex amplitude
        freq_new = img_amp * np.exp(1j * phase)
        # inverse fourier transformation
        img_plane = ifftshift(ifft2(fftshift(freq_new, axes=(1,2)), axes=(1,2)), axes=(1,2)) * cp.exp(-cp_wave_front)
        phase = cp.angle(img_plane) + cp.pi # to save the phase as a pic we need to make it > 0, so we add pi

        # phase quantization
        phase = (phase+cp.pi)/(2*cp.pi)
        phase = ((phase - 1/(2*PHASE_GRAD)) * PHASE_GRAD) # PHASE_GRAD - отвечает за количество уровней квантования (их будет 16)
        phase = (phase.round() * 2 * cp.pi / PHASE_GRAD) - cp.pi

        I = (cp.abs(freq_img) / cp.max(cp.abs(freq_img), axis=0))**2

        # for j in range(10):
            # arr_ssim[j, i] = ssim(I[j], img[j], data_range=1)
            # arr_corr[j, i] = cp.corrcoef(I[j].flatten(), img[j].flatten())[0, 1]
            # arr_nstd[j, i] = NRMSE(I[j], img[j])

    # plotting the phase distribution
    # plt.imshow(phase[0].get(), cmap='gray')
    # plt.show()
    # showing initial pic
    # plt.imshow(np.abs(img[0].get()), cmap='gray')
    # plt.show()
    # showing recreated img
    # plt.imshow(I[0].get(), cmap='gray')
    # plt.show()
    # plotting the ssim values over the iterations
    #plt.plot(list(range(iterations)), arr_ssim)
    #plt.show()

    # print("corr:", np.mean(arr_corr[:,-1]), "+-", np.std(arr_corr[:,-1]))
    # print("nstd:", np.mean(arr_nstd[:,-1]), "+-", np.std(arr_nstd[:,-1]))

    phase_max = cp.max(phase, axis=(1,2), keepdims=True)
    phase = np.uint8((phase.get() / phase_max.get())*255)

    return phase

if __name__=="__main__":
    img_arr = cp.empty((50, N, N))
    holo_arr = cp.empty((50, N, N))
    j = 0

    # test_idx = np.loadtxt("./dataset_64_cifar10/test_indices_25cm.txt").astype(np.int64)

    for i in tqdm(range(100001, 170001)):

        img_arr[j] = cp.asarray(cv2.imread(f'./dataset_32/img/{i}.png', cv2.IMREAD_GRAYSCALE))

        if j == 49:
            holo_arr = phys_gs(img_arr) 

            for k in range(i-49, i+1): 
                cv2.imwrite(f'./dataset_32/holo/{k}.png', holo_arr[49-j])

                # real_holo = np.uint8(np.real(np.exp(1j*(holo_arr[99-j]/255.0*2*np.pi-np.pi)))*255)
                # im_holo = np.uint8(np.imag(np.exp(1j*(holo_arr[99-j]/255.0*2*np.pi-np.pi)))*255)
                
                # cv2.imwrite(f'./dataset_32/holo_512_Re/{k}.png', real_holo)
                # cv2.imwrite(f'./dataset_32/holo_512_Im/{k}.png', im_holo)

                # print(np.corrcoef(holo_arr[99-j].flatten(), initial_phase[0].flatten()))

                # plt.imshow(np.abs((fftshift(fft2(ifftshift(np.exp(1j*(np.float64(holo_arr[99-j]) * 2*np.pi / 255.0 - np.pi)) * np.exp(wave_front))))).get())**2, cmap='gray')
                # # plt.imshow(holo_arr[99-j], cmap='gray')
                # plt.show()

                j -=1

        j+=1
