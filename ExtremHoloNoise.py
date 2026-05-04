import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from numpy import pi, exp
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.io import loadmat, savemat
import time
from numpy import log10, sqrt
from skimage.util import random_noise as imnoise
from skimage.metrics import normalized_root_mse as nrmse_f
import os


def off_axis_image_create(images):
    Nd1 = np.shape(images)[0]
    Np1 = np.shape(images)[1]
    Np2 = int(Np1 * 3.5)
    Np3 = int(Np1 * 4)
    back = np.zeros((Nd1, Np2, Np2))

    for plane in range(Nd1):
        if plane % 4 == 0:
            back[plane, 0:Np1, 0:Np1] = images[plane]
        elif plane % 4 == 1:
            back[plane, 0:Np1, Np2 - Np1:Np2] = images[plane]
        elif plane % 4 == 2:
            back[plane, Np2 - Np1:Np2, Np2 - Np1:Np2] = images[plane]
        else:
            back[plane, Np2 - Np1:Np2, 0:Np1] = images[plane]

    off_axis_images = np.zeros((Nd1, Np3, Np3))
    off_axis_images[:, (Np3 - Np2) // 2:(Np3 - Np2) // 2 + Np2, (Np3 - Np2) // 2:(Np3 - Np2) // 2 + Np2] = back

    return off_axis_images

def img_create(*names, format='.tiff', off_axis=False, show=False):
    images = []
    for name in names:
        path_im = path_base + name + format
        image = cv2.imread(path_im, cv2.IMREAD_GRAYSCALE)
        image = image / np.max(image)
        images.append(image)
    images = np.array(images)
    if off_axis:
        images = off_axis_image_create(images)
    if show:
        for im in images:
            plt.imshow(im)
            plt.show()
    return images

def img_unpack(images):
    Nd1 = np.shape(images)[0]
    Np3 = np.shape(images)[1]
    Np1 = Np3 // 4
    Np2 = int(Np1 * 3.5)
    images_only = np.zeros((Nd1, Np1, Np1))
    back = images[:, (Np3 - Np2) // 2:(Np3 - Np2) // 2 + Np2, (Np3 - Np2) // 2:(Np3 - Np2) // 2 + Np2]
    for plane in range(Nd1):
        if plane % 4 == 0:
            images_only[plane] = back[plane, 0:Np1, 0:Np1]
        elif plane % 4 == 1:
            images_only[plane] = back[plane, 0:Np1, Np2 - Np1:Np2]
        elif plane % 4 == 2:
            images_only[plane] = back[plane, Np2 - Np1:Np2, Np2 - Np1:Np2]
        else:
            images_only[plane] = back[plane, Np2 - Np1:Np2, 0:Np1]
    for index in range(Nd1):
        im = images_only[index]
        im = im / np.max(im)
        images_only[index] = im
    return images_only

def holo_create_Frensel(images, obj_type, off_axis=False):
    holo = np.zeros((Np, Np))
    if obj_type == 'amp':
        if ph_m:
            phase_mask = 2 * pi * (np.random.rand(Nd, Np, Np) - 0.5)
            input_images = images * exp(1j * phase_mask)
        else:
            input_images = images
        k = 2 * pi / wl
        X, Y = np.meshgrid(np.arange(-Np // 2, Np // 2), np.arange(-Np // 2, Np // 2))
        img_pix = wl * z / (Np * hol_pix)
        u = hol_pix * X
        v = hol_pix * Y
        for plane in range(Nd):
            x = img_pix[plane] * X
            y = img_pix[plane] * Y
            image = input_images[plane]
            z_plane = z[plane]

            c1 = exp(1j * k * z_plane) / (1j * wl * z_plane) * exp(1j * pi * (u ** 2 + v ** 2) / (wl * z_plane))
            c2 = image * exp(1j * pi * (x ** 2 + y ** 2) / (wl * z_plane))
            obj = c1 * fftshift(fft2(ifftshift(c2)))

            ref = sqrt(np.mean(np.abs(obj) ** 2))
            holo = holo + obj + ref
        holo = np.abs(holo) ** 2
        holo = holo / np.max(holo)
    elif obj_type == 'phase':
        if off_axis:
            input_images = off_axis_image_create(np.ones((Nd, Np//4, Np//4))) * exp(1j * (images - 0.5) * pi)
        else:
            input_images = np.ones((Nd, Np, Np)) * exp(1j * (images - 0.5) * pi)
        k = 2 * pi / wl
        X, Y = np.meshgrid(np.arange(-Np // 2, Np // 2), np.arange(-Np // 2, Np // 2))
        img_pix = wl * z / (Np * hol_pix)
        u = hol_pix * X
        v = hol_pix * Y
        for plane in range(Nd):
            x = img_pix[plane] * X
            y = img_pix[plane] * Y
            image = input_images[plane]
            z_plane = z[plane]

            c1 = exp(1j * k * z_plane) / (1j * wl * z_plane) * exp(1j * pi * (u ** 2 + v ** 2) / (wl * z_plane))
            c2 = image * exp(1j * pi * (x ** 2 + y ** 2) / (wl * z_plane))
            obj = c1 * fftshift(fft2(ifftshift(c2)))

            ref = sqrt(np.mean(np.abs(obj) ** 2))
            holo = holo + obj + ref
        holo = np.abs(holo) ** 2
        holo = holo / np.max(holo)
    elif obj_type == 'both':
        input_images = images * exp(1j * (images - 0.5) * pi)
        k = 2 * pi / wl
        X, Y = np.meshgrid(np.arange(-Np // 2, Np // 2), np.arange(-Np // 2, Np // 2))
        img_pix = wl * z / (Np * hol_pix)
        u = hol_pix * X
        v = hol_pix * Y
        for plane in range(Nd):
            x = img_pix[plane] * X
            y = img_pix[plane] * Y
            image = input_images[plane]
            z_plane = z[plane]

            c1 = exp(1j * k * z_plane) / (1j * wl * z_plane) * exp(1j * pi * (u ** 2 + v ** 2) / (wl * z_plane))
            c2 = image * exp(1j * pi * (x ** 2 + y ** 2) / (wl * z_plane))
            obj = c1 * fftshift(fft2(ifftshift(c2)))

            ref = sqrt(np.mean(np.abs(obj) ** 2))
            holo = holo + obj + ref
        holo = np.abs(holo) ** 2
        holo = holo / np.max(holo)
    else:
        print('-----------------------------FALSE OBJECT TYPE-----------------------------')
    return holo

def holo_rec_Frensel(holo, obj_type):
    images_rec = np.zeros((Nd, Np, Np))
    if obj_type == 'amp':
        holo = holo / np.max(holo)
        k = 2 * pi / wl
        X, Y = np.meshgrid(np.arange(-Np // 2, Np // 2), np.arange(-Np // 2, Np // 2))
        img_pix = wl * z / (Np * hol_pix)
        u = hol_pix * X
        v = hol_pix * Y
        for plane in range(Nd):
            x = img_pix[plane] * X
            y = img_pix[plane] * Y
            z_plane = - z[plane]

            c1 = exp(1j * k * z_plane) / (1j * wl * z_plane) * exp(1j * pi * (x ** 2 + y ** 2) / (wl * z_plane))
            c2 = holo * exp(1j * pi * (u ** 2 + v ** 2) / (wl * z_plane))
            obj = c1 * ifftshift(ifft2(fftshift(c2)))
            image_rec = np.abs(obj)
            images_rec[plane] = image_rec / np.max(image_rec)
    elif obj_type == 'phase':
        holo = holo / np.max(holo)
        k = 2 * pi / wl
        X, Y = np.meshgrid(np.arange(-Np // 2, Np // 2), np.arange(-Np // 2, Np // 2))
        img_pix = wl * z / (Np * hol_pix)
        u = hol_pix * X
        v = hol_pix * Y
        for plane in range(Nd):
            x = img_pix[plane] * X
            y = img_pix[plane] * Y
            z_plane = - z[plane]

            c1 = exp(1j * k * z_plane) / (1j * wl * z_plane) * exp(1j * pi * (x ** 2 + y ** 2) / (wl * z_plane))
            c2 = holo * exp(1j * pi * (u ** 2 + v ** 2) / (wl * z_plane))
            obj = c1 * ifftshift(ifft2(fftshift(c2)))
            image_rec = np.angle(obj)
            images_rec[plane] = image_rec / np.max(image_rec)
    else:
        print('-----------------------------FALSE OBJECT TYPE-----------------------------')
    return images_rec

def holo_create_Fourier(images, obj_type, off_axis=False):
    holo = np.zeros((Np, Np))
    if obj_type == 'amp':
        if ph_m:
            phase_mask = 2 * pi * (np.random.rand(Nd, Np, Np) - 0.5)
            input_images = images * exp(1j * phase_mask)
        else:
            input_images = images
        X, Y = np.meshgrid(np.arange(-Np // 2, Np // 2), np.arange(-Np // 2, Np // 2))
        img_pix = wl * z / (Np * hol_pix)
        for plane in range(Nd):
            x = img_pix[plane] * X
            y = img_pix[plane] * Y
            image = input_images[plane]
            z_plane = z[plane]

            # c2 = image
            c2 = image * exp(1j * pi * z_plane * (x ** 2 + y ** 2) / (wl * f ** 2))
            obj = fftshift(fft2(ifftshift(c2)))

            ref = sqrt(np.mean(np.abs(obj) ** 2))
            holo = holo + obj + ref
        holo = np.abs(holo) ** 2
        holo = holo / np.max(holo)
    elif obj_type == 'phase':
        if off_axis:
            input_images = off_axis_image_create(np.ones((Nd, Np // 4, Np // 4))) * exp(1j * (images - 0.5) * pi)
        else:
            input_images = np.ones((Nd, Np, Np)) * exp(1j * (images - 0.5) * pi)
        X, Y = np.meshgrid(np.arange(-Np // 2, Np // 2), np.arange(-Np // 2, Np // 2))
        img_pix = wl * z / (Np * hol_pix)
        for plane in range(Nd):
            x = img_pix[plane] * X
            y = img_pix[plane] * Y
            image = input_images[plane]
            z_plane = z[plane]

            # c2 = image
            c2 = image * exp(1j * pi * z_plane * (x ** 2 + y ** 2) / (wl * f ** 2))
            obj = fftshift(fft2(ifftshift(c2)))

            ref = sqrt(np.mean(np.abs(obj) ** 2))
            holo = holo + obj + ref
        holo = np.abs(holo) ** 2
        holo = holo / np.max(holo)
    elif obj_type == 'both':
        input_images = images * exp(1j * (images - 0.5) * pi)
        X, Y = np.meshgrid(np.arange(-Np // 2, Np // 2), np.arange(-Np // 2, Np // 2))
        img_pix = wl * z / (Np * hol_pix)
        for plane in range(Nd):
            x = img_pix[plane] * X
            y = img_pix[plane] * Y
            image = input_images[plane]
            z_plane = z[plane]

            # c2 = image
            c2 = image * exp(1j * pi * z_plane * (x ** 2 + y ** 2) / (wl * f ** 2))
            obj = fftshift(fft2(ifftshift(c2)))

            ref = sqrt(np.mean(np.abs(obj) ** 2))
            holo = holo + obj + ref
        holo = np.abs(holo) ** 2
        holo = holo / np.max(holo)
    else:
        print('-----------------------------FALSE OBJECT TYPE-----------------------------')
    return holo

def holo_rec_Fourier(holo, obj_type):
    images_rec = np.zeros((Nd, Np, Np))
    if obj_type == 'amp':
        holo = holo / np.max(holo)
        for plane in range(Nd):
            obj = ifftshift(ifft2(fftshift(holo)))
            image_rec = np.abs(obj)
            images_rec[plane] = image_rec / np.max(image_rec)
    elif obj_type == 'phase':
        holo = holo / np.max(holo)
        for plane in range(Nd):
            obj = ifftshift(ifft2(fftshift(holo)))
            image_rec = np.angle(obj)
            images_rec[plane] = image_rec / np.max(image_rec)
    return images_rec

def holo_create_Hartley(images, obj_type, off_axis=False):
    holo = np.zeros((Np, Np))
    if obj_type == 'amp':
        if ph_m:
            phase_mask = 2 * pi * (np.random.rand(Nd, Np, Np) - 0.5)
            input_images = images * exp(1j * phase_mask)
        else:
            input_images = images
        for plane in range(Nd):
            image = input_images[plane]
            obj = fftshift(fft2(ifftshift(image)))
            holo = holo + np.real(obj) - np.imag(obj)
        holo = holo - np.min(holo)
        holo = holo / np.max(holo)
    elif obj_type == 'phase':
        if off_axis:
            input_images = off_axis_image_create(np.ones((Nd, Np // 4, Np // 4))) * exp(1j * (images - 0.5) * pi)
        else:
            input_images = np.ones((Nd, Np, Np)) * exp(1j * (images - 0.5) * pi)
        for plane in range(Nd):
            image = input_images[plane]
            obj = fftshift(fft2(ifftshift(image)))
            holo = holo + np.real(obj) - np.imag(obj)
        holo = holo - np.min(holo)
        holo = holo / np.max(holo)
    elif obj_type == 'both':
        input_images = images * exp(1j * (images - 0.5) * pi)
        for plane in range(Nd):
            image = input_images[plane]
            obj = fftshift(fft2(ifftshift(image)))
            holo = holo + np.real(obj) - np.imag(obj)
        holo = holo - np.min(holo)
        holo = holo / np.max(holo)
    else:
        print('-----------------------------FALSE OBJECT TYPE-----------------------------')
    return holo

def holo_rec_Hartley(holo, obj_type):
    images_rec = np.zeros((Nd, Np, Np))
    if obj_type == 'amp':
        holo = holo / np.max(holo)
        for plane in range(Nd):
            obj = ifftshift(ifft2(fftshift(holo)))
            image_rec = np.abs(obj)
            images_rec[plane] = image_rec / np.max(image_rec)
    elif obj_type == 'phase':
        holo = holo / np.max(holo)
        for plane in range(Nd):
            obj = ifftshift(ifft2(fftshift(holo)))
            image_rec = np.angle(obj)
            images_rec[plane] = image_rec / np.max(image_rec)
    return images_rec

def k_calculation_w(holo):
    holo = np.reshape(holo, (np.size(holo)))
    holo = np.sort(holo)
    holo = holo[::-1]
    k = []
    for index in range(np.size(holo)-1):
        m = (holo[index] + holo[index+1]) / 2
        k1 = 1 / m
        k.append(k1)
    return k

def k_calculation_b(holo):
    holo = np.reshape(holo, (np.size(holo)))
    holo = np.sort(holo)
    k = []
    for index in range(np.size(holo)-1):
        m = (holo[index] + holo[index+1]) / 2
        k1 = 2 * m
        k.append(k1)
    return k

def white_range(holo, path):
    k = k_calculation_w(holo)
    p = int(0.01 * np.size(holo))
    holos = []
    for ind in range(0, len(k), p):
        holo_w = np.clip(holo * k[ind], a_min=0, a_max=1)
        holos.append(holo_w)
    return np.array(holos)

def black_range(holo, path):
    holo = holo * (2**bit - 1)
    k = k_calculation_b(holo)
    p = int(0.01 * np.size(holo))
    holos = []
    h_max = []
    for ind in range(0, len(k), p):
        holo_b = np.clip(holo / k[ind], a_min=0.5, a_max=2**bit-1) - 0.5
        h_max.append(np.max(holo_b))
        holo_b = holo_b / np.max(holo_b)
        holos.append(holo_b)
    return np.array(holos), np.array(h_max)

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

# def metrics_calculation(holos, images, holo_rec):
#     ssim = []
#     cc = []
#     psnr = []
#     nrmse = []
#     for percent in range(10):
#         holo = holos[percent]
#         images_rec = holo_rec(holo)
#         images_rec = img_unpack(images_rec)
#         print(percent)
#         ssim_Nd = []
#         cc_Nd = []
#         psnr_Nd = []
#         nrmse_Nd = []
#         for ind in range(Nd):
#             orig = images[ind]
#             rec = images_rec[ind]
#             ssim_Nd.append(calculate_ssim(rec, orig))
#             cc_Nd.append(np.corrcoef(rec.flatten(), orig.flatten()))
#             psnr_Nd.append(cv2.PSNR(rec, orig))
#             nrmse_Nd.append(nrmse_f(orig, rec))
#         ssim.append(float(np.mean(np.array(ssim_Nd))))
#         cc.append(float(np.mean(np.array(cc_Nd))))
#         psnr.append(float(np.mean(np.array(psnr_Nd))))
#         nrmse.append(float(np.mean(np.array(nrmse_Nd))))
#     ssim = np.array(ssim)
#     cc = np.array(cc)
#     psnr = np.array(psnr)
#     nrmse = np.array(nrmse)
#     metr = np.stack((ssim, cc, psnr, nrmse))
#     metr = np.round(metr, 2)
#
#     return metr

def metr_c(images_rec, images):
    ssim_Nd = []
    cc_Nd = []
    psnr_Nd = []
    nrmse_Nd = []
    for ind in range(Nd):
        orig = images[ind]
        rec = images_rec[ind]
        ssim_Nd.append(calculate_ssim(rec*255, orig*255))
        cc_Nd.append(np.corrcoef(rec.flatten(), orig.flatten()))
        psnr_Nd.append(cv2.PSNR(rec*255, orig*255))
        nrmse_Nd.append(nrmse_f(orig, rec))
    ssim = float(np.mean(np.array(ssim_Nd)))
    cc = float(np.mean(np.array(cc_Nd)))
    psnr = float(np.mean(np.array(psnr_Nd)))
    nrmse = float(np.mean(np.array(nrmse_Nd)))
    metr = np.stack((ssim, cc, psnr, nrmse))
    # metr = np.round(metr, 2)

    return metr

def shading(No, r):
    gauss = np.ones([Np, Np])
    x,y = np.meshgrid(np.arange(0, Np), np.arange(0, Np))
    for index in range(0, No):
        sigma1 = (np.random.rand(1) * r*0.5 + r/1.5)
        sigma2 = (np.random.rand(1) * r*0.5 + r/1.5)
        sigma = (sigma1 + sigma2) / 2
        norm_coeff = sigma / (r*0.5 + r / 1.5)

        x_shift = np.random.randint(Np, size=1)
        y_shift = np.random.randint(Np, size=1)
        gauss_new = exp(-((x - x_shift)**2 / (2 * sigma1**2) + (y - y_shift)**2 / (2 * sigma2**2))**2)
        gauss_new = gauss_new / np.max(gauss_new)
        gauss_new = -gauss_new * norm_coeff + 1
        gauss = gauss * gauss_new
        gauss = gauss / np.max(gauss)
    return gauss

def wigneting():
    X, Y = np.meshgrid(np.arange(-Np // 2, Np // 2) + 0.5, np.arange(-Np // 2, Np // 2) + 0.5)

    sigma = float(Np / 5)
    G = exp(-(X ** 2 / (2 * sigma ** 2) + Y ** 2 / (2 * sigma ** 2)))
    G = G / np.max(G)
    C = (X ** 2 + Y ** 2) < (Np / 2) ** 2
    G = (1 - C) * G
    G = G / np.max(G)
    W = C + G
    W = np.clip(W, 0, 1)
    return W

def Spatial_parameters(Ny,Nx,BLO,PRNU,DSNU_DN):
    DSNU = DSNU_DN / BLO
    DARK_DSNU = imnoise(0.5 * np.ones((Ny, Nx)), 'gaussian', mean=0, var=(DSNU) ** 2) + 0.5
    LIGHT_PRNU = imnoise(0.5 * np.ones((Ny, Nx)), 'gaussian', mean=0, var=(PRNU) ** 2) + 0.5
    W = wigneting()
    SH = shading(3, Np/50)
    return SH, W, DARK_DSNU, LIGHT_PRNU

def holo_noise(holo):
    if wig and shd:
        LIGHT = holo * SH * W
    elif wig:
        LIGHT = holo * W
    elif shd:
        LIGHT = holo * SH
    else:
        LIGHT = holo
    LIGHT = LIGHT / np.max(LIGHT)
    # Speckle_coef = np.random.rand(1) * 50
    Speckle_coef = 2
    LIGHT = imnoise(LIGHT, 'speckle', var=Speckle_coef * np.mean(LIGHT)) * (2**bit - 1)
    LIGHT_electr = LIGHT / np.max(LIGHT) * SignalLVL * K

    LIGHT_POISSON = (imnoise((LIGHT_electr / 1e6), 'poisson') * 1e6)
    LIGHT = (LIGHT_POISSON / K) * LIGHT_PRNU


    DARK_TEMP = BLO * (imnoise((0.5 * np.ones((Ny, Nx))), 'gaussian', mean=0, var=(SigmaDT / BLO) ** 2) - 0.5) + BLO
    DARK = DARK_TEMP * DARK_DSNU
    LIGHT_FRAME = LIGHT + DARK
    LIGHT_FRAME = LIGHT_FRAME / np.max(LIGHT_FRAME)

    return LIGHT_FRAME

def minmax_show(arr):
    print('Минимальное значение массива: ' + str(np.min(arr)))
    print('Среднее значение массива: ' + str(np.mean(arr)))
    print('Максимальное значение массива: ' + str(np.max(arr)))


# --------------------------------------------------Starting parametrs--------------------------------------------------
path_base = 'D:\\Study\\Science\\Noise\\'
# path_base = 'D:\\Study\\Science\\Interpol_filter_article\\images\\'
# path_base = 'D:\\Study\\Science\\train-test\\numbers_sint_p1_5z0_128x128\\test_images\\'
ph_m = True
off_axis_param = True
wig = False
shd = False
# images = img_create('4.2.07', '4.2.03', '4.2.07', '4.2.03', off_axis=True, show=True)
# images = img_create('4.2.07', '4.2.03', off_axis=True, show=True)
# images = img_create('4.2.07', off_axis=True, show=True)
# images_true = img_unpack(images)

Nd = 1024
Np = 1024
hol_pix = 8e-6
bit = 14
wl = 532e-9
nz0 = 10
zstep = 1
f = 0.3


z0 = (Np * hol_pix ** 2) / wl
zstart = nz0 * z0
z = np.arange(Nd) * zstep + zstart

# ------------------------------------------------------ Param.m -------------------------------------------------------

# Задать характеристики камеры
Nx=Np
Ny=Np
bit=bit

BLO = 100
K = 1.2
PRNU = 0.0034       # В долях
DSNU_DN = 0.5       # В цифровых единицах
DSNU = 5 / BLO
SigmaDT = 4.46
SignalLVL = (2 ** bit) / 2 - BLO
DinamicRange = 20 * log10((2 ** bit - BLO) / SigmaDT)
Bar_number = 4

SH, W, DARK_DSNU, LIGHT_PRNU = Spatial_parameters(Ny,Nx,BLO,PRNU,DSNU_DN)

# ----------------------------------------------------------------------------------------------------------------------

for i in range(100001, 170000):

    img = cv2.imread(f"dataset_32/img/{i}.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    img1 = np.zeros((1024, 1024))
    img1[:img.shape[0], :img.shape[1]] = img

    holo = holo_create_Frensel(img1, img.dtype, off_axis=True)

    plt.imshow(holo, cmap='gray')
    plt.show()

    break

# t1 = time.time()

# fold = 'Hartley\\Peppers\\'

# if not (os.path.exists(path_base + fold)):
#     os.mkdir(path_base + fold)

# for i in range(Nd):
#     cv2.imwrite(path_base + fold + 'img_' + str(i) + '.png', (images_true[i] * 255).astype(np.uint8))

# holo = holo_create_Hartley(images, 'amp', off_axis=off_axis_param)
# cv2.imwrite(path_base + fold + 'holo.png', (holo * (2**bit - 1)).astype(np.uint16))

# noised_holo = holo_noise(holo)
# cv2.imwrite(path_base + fold + 'holo_noise.png', (noised_holo * (2**bit - 1)).astype(np.uint16))

# holos_b, h_max = black_range(noised_holo, '')
# metrics = []
# counter = 0

# if not (os.path.exists(path_base + fold + 'b\\')):
#     os.mkdir(path_base + fold + 'b\\')

# if not (os.path.exists(path_base + fold + 'b\\holos\\')):
#     os.mkdir(path_base + fold + 'b\\holos\\')

# if not (os.path.exists(path_base + fold + 'b\\rec_images\\')):
#     os.mkdir(path_base + fold + 'b\\rec_images\\')

# for h in holos_b:

#     holo_new = h
#     percent = np.sum(holo_new==0) / np.size(holo_new)
#     print(percent)

#     cv2.imwrite(path_base + fold + 'b\\holos\\' + str(int(np.round(percent*100, 0))) + '%.png', np.ceil(holo_new * h_max[counter]).astype(np.uint16))

#     rec_noise_images = holo_rec_Hartley(holo_new * h_max[counter], 'amp')
#     rec_noise_images_true = img_unpack(rec_noise_images)

#     for i in range(Nd):
#         cv2.imwrite(path_base + fold + 'b\\rec_images\\' + 'rec_noise_' + str(int(np.round(percent*100, 0))) + '%_img_' + str(i) + '.png', (rec_noise_images_true[i] * 255).astype(np.uint8))

#     m = metr_c(rec_noise_images_true, images_true)
#     m = np.insert(m, 0, np.round(percent*100,0))
#     print(m)
#     metrics.append(m)
#     counter = counter + 1

# metrics = pd.DataFrame(metrics,
#                        columns=['percent', 'ssim', 'cc', 'psnr', 'nrmse'], )
# print(np.array(metrics))
# metrics.to_csv(path_base + fold + 'b\\' + 'metrics.csv')

# # f1, [p1, p2] = plt.subplots(1, 2)
# # p1.imshow(holo)
# # p1.axis('off')
# # p1.title.set_text('holo_orig')
# # p2.imshow(holo_new)
# # p2.axis('off')
# # p2.title.set_text('holo_noise')
# # plt.savefig(path_base + 'Hartley_holos.png', dpi=400)
# # plt.show()

# # for ind in range(Nd):
# #     f, (a1, a2, a3) = plt.subplots(1, 3)
# #     a1.imshow(images_true[ind])
# #     a2.imshow(rec_images_true[ind])
# #     a3.imshow(rec_noise_images_true[ind])
# #     a1.axis('off')
# #     a2.axis('off')
# #     a3.axis('off')
# #     a1.title.set_text('orig_image' + ' plane_' + str(ind + 1))
# #     a2.title.set_text('rec_image' + ' plane_' + str(ind + 1))
# #     a3.title.set_text('rec_noise_image' + ' plane_' + str(ind + 1))
# #     plt.savefig(path_base + 'Hartley_images.png', dpi=400)
# #     plt.show()

# # for ind in range(Nd):
# #     f, (a1, a2, a3) = plt.subplots(1, 3)
# #     a1.imshow(images[ind])
# #     a2.imshow(rec_images[ind])
# #     a3.imshow(rec_noise_images[ind])
# #     a1.axis('off')
# #     a2.axis('off')
# #     a3.axis('off')
# #     a1.title.set_text('orig_image' + ' plane_' + str(ind + 1))
# #     a2.title.set_text('rec_image' + ' plane_' + str(ind + 1))
# #     a3.title.set_text('rec_noise_image' + ' plane_' + str(ind + 1))
# #     plt.show()

# holos_w = white_range(noised_holo, '')
# metrics = []

# if not (os.path.exists(path_base + fold + 'w\\')):
#     os.mkdir(path_base + fold + 'w\\')

# if not (os.path.exists(path_base + fold + 'w\\holos\\')):
#     os.mkdir(path_base + fold + 'w\\holos\\')

# if not (os.path.exists(path_base + fold + 'w\\rec_images\\')):
#     os.mkdir(path_base + fold + 'w\\rec_images\\')

# for h in holos_w:

#     holo_new = h
#     percent = np.sum(holo_new==1) / np.size(holo_new)
#     print(percent)

#     cv2.imwrite(path_base + fold + 'w\\holos\\' + str(int(np.round(percent*100, 0))) + '%.png', (holo_new * (2**bit - 1)).astype(np.uint16))

#     rec_noise_images = holo_rec_Hartley(holo_new, 'amp')
#     rec_noise_images_true = img_unpack(rec_noise_images)

#     for i in range(Nd):
#         cv2.imwrite(path_base + fold + 'w\\rec_images\\' + 'rec_noise_' + str(int(np.round(percent*100, 0))) + '%_img_' + str(i) + '.png', (rec_noise_images_true[i] * 255).astype(np.uint8))

#     m = metr_c(rec_noise_images_true, images_true)
#     m = np.insert(m, 0, np.round(percent*100,0))
#     print(m)
#     metrics.append(m)

# metrics = pd.DataFrame(metrics,
#                        columns=['percent', 'ssim', 'cc', 'psnr', 'nrmse'], )
# print(np.array(metrics))
# metrics.to_csv(path_base + fold + 'w\\' + 'metrics.csv')

# t2 = time.time()
# dt = t2 - t1
# print('Время выполнения кода ' + str(dt))