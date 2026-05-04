import imageio.v2 as imag
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

# images = [f'folder_rubbish/___{i}.jpeg' for i in range(10)]
# with imag.get_writer('folder_rubbish/animaton_6.gif', mode='I', duration=1000) as writer:
#     for img in images:
#         writer.append_data(imag.imread(img))

# images2 = [f'2_for_preseint_{i}.jpeg' for i in range(10)]

# with imag.get_writer('animaton_2.gif', mode='I', duration=1000) as writer:
#     for img in images2:
#         writer.append_data(imag.imread(img))

# images = [f'3_for_preseint_{i}.jpeg' for i in range(10)]

# with imag.get_writer('animaton_3.gif', mode='I', duration=1000) as writer:
#     for img in images:
#         writer.append_data(imag.imread(img))







# frame_durations = [0.5,0.5,0.5,0.5,0.5,0.5,0.5]  

# with imageio.get_writer('animation.gif', mode='I') as writer:
#     for img, duration in zip(images, frame_durations):
#         image = imageio.imread(img)
#         writer.append_data(image)

















"""           НЕ ЛЕЗЬ ДАЛЬШЕ, Я УЖЕ САМ НЕ ПОНИМАЮ КУДА ГДЕ И ЧТО               """
images_1 = [cv2.imread(f'_{i}.jpeg', cv2.IMREAD_GRAYSCALE)[:51, :51] for i in range(10)]
images_2 = [cv2.imread(f'___{i}.jpeg', cv2.IMREAD_GRAYSCALE)[:51, :51] for i in range(10)] 
cv2.imshow('dasf', images_1[0], )
cv2.waitKey(0)
arr_ssim = []

for i in range(10):
    arr_ssim.append(ssim(images_1[i]/np.max(images_1[i]), images_2[i]/np.max(images_2[i]), data_range=1, win_size=51))

print(arr_ssim)