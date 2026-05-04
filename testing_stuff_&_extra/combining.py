from pathlib import Path
from os import remove, rmdir
import cv2
import shutil

# for i in range(8000):
#     Path(f"./dataset_512/pair_{i}").mkdir(parents=True, exist_ok=True)
#     img = cv2.imread(f'./numbers/{i}.tiff', cv2.IMREAD_GRAYSCALE)
#     cv2.imwrite(f"./dataset_512/pair_{i}/num_{i}.tiff", img)
#     img = cv2.imread(f'./numbers_holo/{i}.tiff', cv2.IMREAD_GRAYSCALE)
#     cv2.imwrite(f"./dataset_512/pair_{i}/holo_{i}.tiff", img)

for i in range(11000, 20000):
    img = cv2.imread(f"./numbers/{i}.tiff")
    holo = cv2.imread(f"./numbers_holo/{i}.tiff")

    # cv2.imwrite(f"./dataset_512/pair_{i}/num_{i}.jpeg", img)
    # cv2.imwrite(f"./dataset_512/pair_{i}/holo_{i}.jpeg", holo)

    # remove(f"./dataset_512/pair_{i}/num_{i}.tiff")
    # remove(f"./dataset_512/pair_{i}/holo_{i}.tiff")

    # img = cv2.imread(f"./dataset_512/pair_{i}/a_num.jpeg")
    # holo = cv2.imread(f"./dataset_512/pair_{i}/b_holo.jpeg")

    cv2.imwrite(f"./dataset_512/img/a_num_{i}.jpeg", img)
    cv2.imwrite(f"./dataset_512/holo/b_holo_{i}.jpeg", holo)

    # remove(f"./dataset_512/pair_{i}")
    # remove(f"./dataset_512/pair_{i}")

    # rmdir(f"./dataset_512/pair_{i}")
    # rmdir(f"./dataset_512/pair_{i}")

    # shutil.rmtree(f"./dataset_512/pair_{i}")

'''
Если вы хотите удалить файл

import os
os.remove("path_to_file")
но вы не сможете удалить каталог с помощью приведенного выше кода. Если вы хотите удалить каталог, используйте этот код

import os
os.rmdir("path_to_dir")
С помощью приведенной выше команды вы можете удалить каталог, если он пуст. Если он не пуст, вы можете использовать модуль shutil

import shutil
shutil.rmtree("path_to_dir")
Все вышеперечисленные методы являются методами Python, и если вы знаете, что ваша операционная система зависит от этого метода, то все вышеперечисленные методы не зависят от ОС

import os
os.system("rm -rf _path_to_dir")
'''
