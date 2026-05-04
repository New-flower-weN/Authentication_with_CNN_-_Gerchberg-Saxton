import subprocess

for num_of_folders in (2**i for i in range(5)):
    # print(num_of_folders)
    # proc = subprocess.run(["./myenv/Scripts/activate"])
    proc = subprocess.run(["./myenv/Scripts/python", "./Neural_networks/unet.py", f"{num_of_folders}"])