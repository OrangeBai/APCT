
def update_params(plt):
    large = 22
    med = 16
    small = 12
    params = {'axes.titlesize': large,
              'legend.fontsize': large,
              'figure.figsize': (12, 8),
              'axes.labelsize': large,
              'xtick.labelsize': large,
              'ytick.labelsize': large,

              'figure.titlesize': large}
    plt.style.use('ggplot')
    plt.rcParams.update(params)
    return plt