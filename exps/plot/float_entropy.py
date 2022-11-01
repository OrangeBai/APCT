import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    large = 22
    med = 16
    small = 12
    params = {'axes.titlesize': large,
              'legend.fontsize': large,
              'axes.labelsize': large,
              'xtick.labelsize': large,
              'ytick.labelsize': large,

              'figure.titlesize': large}
    xx = np.linspace(1e-5, 1 - 1e-5, 200)
    yy = -xx * np.log(xx) - (1-xx) * np.log(1-xx)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(xx, yy)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylabel('Float Entropy', fontsize=24)
    ax.set_xlabel('Probability of Status 0', fontsize=24)
    plt.show()