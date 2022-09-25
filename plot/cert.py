from plot.cfg import *
import numpy as np

if __name__ == '__main__':
    files = [
        '/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_005/exp/SMRAP_cert.npy',
        '/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_005/exp/STD_cert.npy'
        ]
    fig, ax = plt.subplots()
    for file in files:
        res = np.load(file)
        x = np.linspace(0, 1, len(res))
        ax.plot(x, res)
    ax.legend(['SMRAP', 'STD'])
    plt.show()



