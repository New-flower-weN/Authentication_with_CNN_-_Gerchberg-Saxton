import numpy as np
import matplotlib.pyplot as plt

arr = []

for i in range(100):
    data = np.loadtxt(f"./NN16/{i}phase_corr.txt")
    mean = np.mean(data)
    sigma = np.std(data)
    print(mean, sigma)

    plt.hist(data, bins=30, density=True, color='blue', edgecolor='black')

    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label='mean')

    plt.axvline(mean + sigma, color='green', linestyle='dashed', linewidth=2, label='mean+sigma')
    plt.axvline(mean - sigma, color='green', linestyle='dashed', linewidth=2, label='mean+sigma')

    plt.axvline(mean + 2*sigma, color='orange', linestyle='dashed', linewidth=2, label='mean+2sigma')
    plt.axvline(mean - 2*sigma, color='orange', linestyle='dashed', linewidth=2, label='mean+2sigma')

    plt.axvline(mean + 3*sigma, color='purple', linestyle='dashed', linewidth=2, label='mean+3sigma')
    plt.axvline(mean - 3*sigma, color='purple', linestyle='dashed', linewidth=2, label='mean+3sigma')

    plt.xlabel('Value')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid(True)

    # plt.show()

    arr.append([mean, sigma])

np.savetxt('./NN16/mean_sigma.txt', arr)