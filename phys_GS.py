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

# initial_phase = np.ones((N, N)) * np.pi / 2

# initial_phase =  np.random.uniform(0, 1, (N, N))
# initial_phase = np.ones((N, N)) * 0.5 
# initial_phase = np.ones((N, N)) 
# initial_phase[0:32, 0:32] = cv2.imread('./dataset_64_cifar10/mean_64forthedge_60kNN.png', cv2.IMREAD_GRAYSCALE) / 255.0
# initial_phase[32:64, 32:64] = cv2.imread('./dataset_64_cifar10/mean_64forthedge_60kNN.png', cv2.IMREAD_GRAYSCALE) / 255.0
# initial_phase[0:32, 32:64] = cv2.imread('./dataset_64_cifar10/mean_64forthedge_60kNN.png', cv2.IMREAD_GRAYSCALE) / 255.0
# initial_phase[32:64, 0:32] = cv2.imread('./dataset_64_cifar10/mean_64forthedge_60kNN.png', cv2.IMREAD_GRAYSCALE) / 255.0
# initial_phase = initial_phase * 2 * np.pi - np.pi

# initial_phase = cv2.imread('./dataset_32/phase_1.png', cv2.IMREAD_GRAYSCALE) / 255.0 * 2 * np.pi - np.pi

# initial_phase = 2 * np.pi * cv2.imread('./dataset_64_cifar10/quad_phase.png', cv2.IMREAD_GRAYSCALE) / 255 - np.pi 
# initial_phase = cv2.imread('./dataset_32/phase_1.png', cv2.IMREAD_GRAYSCALE)
# initial_phase = 2 * np.pi * initial_phase / np.max(initial_phase) - np.pi

# initial_phase = np.pi * np.random.uniform(-1, 1, (N, N))
initial_phase = np.random.randint(0, 16, (N, N))
initial_phase = (initial_phase / 16 )*2*np.pi - np.pi
cv2.imwrite('./dataset_32/phase_1.png', np.uint8((initial_phase+np.pi)/(2*np.pi)*255))
initial_phase = np.repeat(initial_phase.reshape((1,N,N)), 50, axis=0)

# initial_phase = np.ones((N, N)) + 0.05 * np.random.randn(N, N)
# cv2.imwrite('./dataset_64x2_cifar10/almost_const_phase4.png', np.uint8((initial_phase+np.pi)/(2*np.pi)*255))

# setting the grid for fourier transformation and implementing the phys. parameters like size of pixel for camera/FTML
X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
# X, Y = np.meshgrid(np.linspace(-N1 / 2, N1 / 2, 256), np.linspace(-N1 / 2, N1 / 2, 256))

du = lmbd * z / (N * ps)
u = ps * X
v = ps * Y
x = du * X
y = du * Y

x2 = du2 * X
y2 = du2 * Y

# for shortening creating var that'll be under exp()
wave_front = (1j * np.pi / (lmbd * z)) * (u * u + v * v)
wave_front2 = (1j * np.pi / (lmbd * z2)) * (u * u + v * v)

cp_wave_front = cp.asarray((1j * np.pi / (lmbd * z)) * (u * u + v * v))

# print("{:.15f}".format ,d_AH[1])
# print("Pi: {:.15f}\n z*l: {:.15f}\n dH*dH: {:.15f}".format(np.pi, lmbd * z, ps*ps))
# print("{:.15f}\n".format(np.exp(wave_front)))
# print(u*u)

# size of arrays and number of iterations for the algorithm
iterations = 100
num_of_pics = 10

# arr_it = np.zeros((1000, 2))
jj = 0

# algorithm for 1 picture only, img values should be <=1
def phys_gs_one(img, hight, width):
    # creating amplitude from intencivity from initial pic & creating initial random phase distribution over the grid 
    # (doesn't matter if it's from -pi to pi or from 0 to 2pi -> algorithm outputs phase ditribution from -pi to pi)
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

# algorithm for 1 picture only, img values should be <=1
def phys_gs_two(img1, img2, hight, width):
    # creating amplitude from intencivity from initial pic & creating initial random phase distribution over the grid 
    # (doesn't matter if it's from -pi to pi or from 0 to 2pi -> algorithm outputs phase ditribution from -pi to pi)

    intens1 = cv2.resize(img1, (60,60))
    intens2 = cv2.resize(img2, (60,60))
    
    img_amp1 = np.zeros((128,128))
    img_amp1[:60, :60] = np.sqrt(intens1)
    img_amp2 = np.zeros((128,128))
    img_amp2[-61:-1, -61:-1] = np.sqrt(intens2)

    # plt.imshow(img_amp2)
    # plt.show()

    # phase1 = np.pi * np.random.uniform(-1, 1, (hight, width)) # np.random.rand(hight, widht)
    # phase2 = np.pi * np.random.uniform(-1, 1, (hight, width))

    phase1 = phase2 = initial_phase

    # arr_ssim = np.zeros(iterations) # metric for compering restored & initial pics

    for i in range(iterations):#tqdm(range(iterations)):
        complex_amp1 = 1 * np.exp(1j * phase1)

        freq_img1 = fftshift(fft2(ifftshift(complex_amp1 * np.exp(wave_front)))) 

        phase1 = np.angle(freq_img1)
        freq_new1 = img_amp1 * np.exp(1j * phase1)
        img_plane1 = ifftshift(ifft2(fftshift(freq_new1))) * np.exp(-wave_front) 
        phase1 = np.angle(img_plane1) 


        complex_amp2 = 1 * np.exp(1j * phase2)

        freq_img2 = fftshift(fft2(ifftshift(complex_amp2 * np.exp(wave_front2)))) 

        phase2 = np.angle(freq_img2)
        freq_new2 = img_amp2 * np.exp(1j * phase2)
        img_plane2 = ifftshift(ifft2(fftshift(freq_new2))) * np.exp(-wave_front2) 
        phase2 = np.angle(img_plane2) 

    # phase = phase1 + phase2 + 2*np.pi
    phase = np.angle(img_plane1+img_plane2) + np.pi #тут фаза от - пи до пи -> + np.pi

    return np.uint8((phase) / (2*np.pi) * 255)

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
    
    '''
    for phys_gs_two
    '''

    # #write holo

    # j = 110001

    # for i in tqdm(range(120001, 122000, 2)):
    #     img1 = cv2.imread(f'./dataset_128/img/{i}.png', cv2.IMREAD_GRAYSCALE) / 255
    #     img2 = cv2.imread(f'./dataset_128/img/{i+1}.png', cv2.IMREAD_GRAYSCALE) / 255

    #     holo = phys_gs_two(img1, img2, 128,128)

    #     cv2.imwrite(f'./dataset_128/holo/{j}.png', holo)

    #     # #read holo
    #     # holo = cv2.imread(f'./dataset_128/holo/{j}.png', cv2.IMREAD_GRAYSCALE) 
        
    #     # res = np.abs(fftshift(fft2(ifftshift(np.exp(1j * (holo / 255 * 2*np.pi - np.pi)) * np.exp(wave_front)))))**2

    #     # plt.imshow(res, cmap='gray')
    #     # plt.show()

    #     j+=1


    '''
    for phys_gs_one
    '''

    # for i in tqdm(range(100001, 170001)):

    #     # holo = cv2.imread(f'./dataset_64/holo_1_dist/{i}.png', cv2.IMREAD_GRAYSCALE)
    #     # # holo = cv2.imread(f'./H_k_64_50_3/K/{i}.png', cv2.IMREAD_GRAYSCALE)
    #     # plt.imshow(holo, cmap='gray')
    #     # plt.show()
    #     # plt.imshow(np.abs(fftshift(fft2(ifftshift(np.exp(1j*holo * 2*np.pi / 255.0) * np.exp(wave_front)))))**2, cmap='gray')
    #     # plt.show()

    #     # Creating hologramms

    #     # img = cv2.imread(f'./dataset_classifier/img_512/{i}.png', cv2.IMREAD_GRAYSCALE) / 255.0
    #     img = cv2.imread('E:/plane128.tif', cv2.IMREAD_GRAYSCALE) / 255.0
    #     # hack_phase = cv2.imread(f'./dataset_64x2_cifar10/holo_all_random/{i}.png', cv2.IMREAD_GRAYSCALE) / 255.0 * 2 * np.pi - np.pi

    #     # plt.imshow(img, cmap='gray')
    #     # plt.show()

    #     # holo = phys_gs_one(img, N, N)

    #     holo = cv2.imread('E:/plane128_holo_bin.bmp', cv2.IMREAD_GRAYSCALE)
    #     # holo = cv2.imread(f'./dataset_classifier/experiment/{i}.png', cv2.IMREAD_GRAYSCALE)
    #     # print(holo)
    #     # holo = holo / np.max(holo)

    #     # temp_phase = holo
    #     # temp_initial_phase = initial_phase

    #     # # Нормализуем данные
    #     # X = temp_initial_phase.flatten()
    #     # Y = temp_phase.flatten()
    #     # # print(np.std(X), np.std(Y))
    #     # # print('mean:', np.mean(X), np.mean(Y))

    #     # X = (X - np.mean(X)) / np.std(X)
    #     # Y = (Y - np.mean(Y)) / np.std(Y)

    #     # # Ортогонализация: Y_orth = Y - proj_X(Y)
    #     # proj = np.dot(Y, X) / np.dot(X, X)
    #     # Y_orth = Y - proj * X
    #     # Y_orth = Y_orth.reshape((128, 128))
    #     # Y_orth = (Y_orth - np.min(Y_orth)) / (np.max(Y_orth) - np.min(Y_orth))
    #     # Y_orth = np.uint8(Y_orth * 255)

    #     # print(f'11 iter {i}:', np.corrcoef(Y_orth.flatten(), Y.flatten())[0, 1])
    #     # print(f'22 iter {i}:', np.corrcoef(Y_orth.flatten(), X.flatten())[0, 1])

    #     # plt.imshow(holo, cmap='gray')
    #     # plt.show()

    #     # res = np.abs(fftshift(fft2(ifftshift(np.exp(1j*(np.float64(holo) * 2*np.pi / 255.0 - np.pi)) * np.exp(wave_front)))))**2
    #     # res = res / np.max(res)
    #     # res = np.uint8(res*255)

    #     plt.imshow(np.abs(fftshift(fft2(ifftshift(np.exp(1j*(np.float64(holo) * 2*np.pi / 255.0 - np.pi)) * np.exp(wave_front)))))**2, cmap='gray')
    #     plt.show()

    #     # hologram = np.angle(ifftshift(ifft2(fftshift(np.sqrt(img)*np.exp(1j*holo * 2*np.pi / 255.0 - 1j*np.pi)))) * np.exp(-wave_front))
    #     # plt.imshow(np.abs(fftshift(fft2(ifftshift(np.exp(1j*(hologram - np.pi)) * np.exp(wave_front)))))**2, cmap='gray')
    #     # plt.show()

    #     # print(1j*(holo * 2*np.pi / 255.0 - np.pi)[0], (1j*holo * 2*np.pi / 255.0 - 1j*np.pi)[0], sep='\n\n')
    #     # print((1j*holo * 2*np.pi / 255.0)[0] + (holo * 2*np.pi / 255.0)[0])

    #     # cv2.imwrite(f'./dataset_classifier/experiment/{i}.bmp', holo)
    #     cv2.imwrite('E:/plane128_holo.bmp', holo)
    #     # break
    #     # print(np.corrcoef(initial_phase.flatten(), holo.flatten())[0][1])
    #     # cv2.imwrite(f'./dataset_64x2_cifar10/phases_all_random/{i}.png', initial_phase)
    # # np.savetxt('arr_it.txt', arr_it)

    '''
    for phys_gs
    '''

    # print(f"CUDA available: {cp.cuda.is_available()}")

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
