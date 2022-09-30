from core.smooth_analyze import *

if __name__ == '__main__':
    large = 22
    med = 16
    small = 12
    params = {'axes.titlesize': large,
              'legend.fontsize': large,
              'figure.figsize': (16, 9),
              'axes.labelsize': large,
              'xtick.labelsize': large,
              'ytick.labelsize': large,

              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-dark')
    sns.set_theme(style="darkgrid")

    files = [
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_025/exp/STD_100_10000_0.25_0',
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_050/exp/STD_100_10000_0.5_0.0',
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_100/exp/STD_100_10000_1.0_0.0',

        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_025/exp/SMRAP_100_10000_0.25_-0.1',
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_050/exp/SMRAP_100_10000_0.5_-0.1',
        r'/home/orange/Main/Experiment/ICLR/cifar10/vgg16_noise_100/exp/SMRAP_100_10000_1.0_-0.1',
    ]
    line_styles = ['r-',  'g-', 'b-', 'r--', 'g--', 'b--']
    fig, ax = plt.subplots()
    ress = []
    for file, line_style in zip(files, line_styles):
        res = ApproximateAccuracy(file).at_radii(np.linspace(0, 3, 60))
        ress += [res]
        # res = np.load(file)
        x = np.linspace(0, 3, len(res))
        ax.plot(x, res, line_style)
        s = ''
        for i in range(0, 60, 5):
            s += '{0:.1f}'.format(res[i] * 100) + '&'
        print(s)
    ax.legend(['Cohen et al ($\sigma=0.25$)',
               'Cohen et al ($\sigma=0.50$)',
               'Cohen et al ($\sigma=1.00$)',
               'Cohen et al + SCRFP ($\sigma=0.20)$',
               'Cohen et al + SCRFP ($\sigma=0.50)$',
               'Cohen et al + SCRFP ($\sigma=1.00)$',
               ], fontsize=18)
    # ax.set_xlim([0, 2.5])
    # ax.set_ylim([0.10, 0.40])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    ax.set_xlabel('Radius', fontsize=24)
    ax.set_ylabel('Accuracy', fontsize=24)
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    # plt.show()
    ress = np.array(ress)
    plt.savefig('cifar.png', bbox_inches='tight')
    print(1)
