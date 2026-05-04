import subprocess
import numpy as np

for num_of_folders in (2**i for i in range(1, 5)):
    # print(num_of_folders)
    # proc = subprocess.run(["./myenv/Scripts/activate"])
    proc = subprocess.run(["./myenv/Scripts/python", "./corr_memeff.py", f"{num_of_folders}"])

    # arr = np.loadtxt(f"./{num_of_folders}phase_corr.txt")
    # print(f"{num_of_folders}: ", np.mean(arr[:, 0]), np.max(arr[:, 1]), np.min(arr[:, 2]))