from plot.cfg import *
import numpy as np
from core.smooth_analyze import *
if __name__ == '__main__':
    files = [
        # '/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_005/exp/SMRAP_cert.npy',
        # '/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_005/exp/STD_cert.npy'
        r'/home/user/Orange/ICLR/imagenet/resnet50_noise_000/exp/SMRAP_50_10000_0.25_-0.05',
        r'/home/user/Orange/ICLR/imagenet/resnet50_noise_000/exp/STD_50_10000_0.25_0.0'
        ]
    fig, ax = plt.subplots()



    for file in files:
        res = ApproximateAccuracy(file).at_radii(np.linspace(0, 1, 256))
        # res = np.load(file)
        x = np.linspace(0, 1, len(res))
        ax.plot(x, res)
    ax.legend(['SMRAP', 'STD'])
    plt.show()



