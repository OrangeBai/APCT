from plot.cfg import *
import numpy as np
from core.smooth_analyze import *

if __name__ == '__main__':
    files = [
        '/home/orange/Main/Experiment/ICLR/exp/SMRAP_100_1000_0.25_-0.05',
        '/home/orange/Main/Experiment/ICLR/exp/SMRAP_100_1000_0.25_-0.1',
        '/home/orange/Main/Experiment/ICLR/exp/SMRAP_100_1000_0.25_-0.25',
        '/home/orange/Main/Experiment/ICLR/exp/STD_100_1000_0.25_0.0',
        '/home/orange/Main/Experiment/ICLR/exp/STD_100_1000_0.25_1.0',
        
        '/home/orange/Main/Experiment/ICLR/exp/SMRAP_100_1000_0.5_-0.05',
        '/home/orange/Main/Experiment/ICLR/exp/SMRAP_100_1000_0.5_-0.1',
        '/home/orange/Main/Experiment/ICLR/exp/SMRAP_100_1000_0.5_-0.25',
        '/home/orange/Main/Experiment/ICLR/exp/STD_100_1000_0.5_0.0',
        '/home/orange/Main/Experiment/ICLR/exp/STD_100_1000_0.5_1.0',
        
        '/home/orange/Main/Experiment/ICLR/exp/SMRAP_100_1000_1.0_-0.05',
        '/home/orange/Main/Experiment/ICLR/exp/SMRAP_100_1000_1.0_-0.1',
        '/home/orange/Main/Experiment/ICLR/exp/SMRAP_100_1000_1.0_-0.25',
        '/home/orange/Main/Experiment/ICLR/exp/STD_100_1000_1.0_0.0',
        '/home/orange/Main/Experiment/ICLR/exp/STD_100_1000_1.0_1.0'
    ]
    fig, ax = plt.subplots()

    res = []
    for file in files:
        cur = ApproximateAccuracy(file).at_radii(np.linspace(0, 2.7, 300))
        # res = np.load(file)
        x = np.linspace(0, 3, len(cur))
        ax.plot(x, cur)
        res.append(cur)
    ax.legend(['SMRAP_1', 'SMRAP_2', 'SMRAP_3', 'Samon et al', 'Cohen et al'])
    # ax.set_ylim(0.3, 0.6)
    plt.show()
