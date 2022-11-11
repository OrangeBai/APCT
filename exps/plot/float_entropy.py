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
    # ReLU
    yy = -xx * np.log(xx) - (1-xx) * np.log(1-xx)

    # Sigmoid
    p2 = (1 - xx) / 2
    yy2 = -xx * np.log(xx) - 2 * p2 * np.log(p2)

    # Sigmoid
    p2 = (1 - xx) / 3
    yy3 = -xx * np.log(xx) - 3 * p2 * np.log(p2)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(xx, yy, label='Activation with 2 regions')
    ax.plot(xx, yy2, 'orange', label='Activation with 3 regions')
    ax.plot(xx, yy3, 'green', label='Activation with 4 regions')
    ax.fill_between(xx, yy2, yy, color='orange', alpha=0.2)
    ax.fill_between(xx, yy3, yy2, color='green', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylabel('Float Entropy', fontsize=24)
    ax.set_xlabel('Probability of Pattern 0', fontsize=24)
    fig.legend(fontsize=24)
    plt.show()
    print(1)
