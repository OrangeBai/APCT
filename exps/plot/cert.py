import matplotlib.pyplot as plt

from plot.cfg import *
from core.smooth_analyze import *

if __name__ == '__main__':
    files = [
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_025/exp/SMRAP_100_10000_0.05',
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_025/exp/SMRAP_100_10000_0.1',
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_025/exp/SMRAP_100_10000_0.25',
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_025/exp/STD_100_10000_0.05',
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_025/exp/STD_100_10000_0.1',
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_025/exp/STD_100_10000_0.25'
    ]
    line_styles = ['r-', 'g-', 'b-', 'r--', 'g--', 'b--', ]
    fig, ax = plt.subplots()
    for file, line_style in zip(files, line_styles):
        res = ApproximateAccuracy(file).at_radii(np.linspace(0, 1, 100))
        # res = np.load(file)
        x = np.linspace(0, 1, len(res))
        ax.plot(x, res, line_style)
    ax.legend(['RSRAP-$\sigma=0.05$', 'RSRAP-$\sigma=0.10$', 'RSRAP-$\sigma=0.25$',
               'RS-$\sigma=0.05$', 'RS-$\sigma=0.10$', 'RS-$\sigma=0.25$'], fontsize=18)
    ax.set_xlim([0, 0.85])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    ax.set_xlabel('Radius')
    ax.set_ylabel('Accuracy')
    plt.savefig('cert-025.png', bbox_inches='tight')
