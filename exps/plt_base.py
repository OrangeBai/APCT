
def update_params(plt, new_params=None):
    plt.style.use('ggplot')
    params = {'axes.titlesize': 16,
              'legend.fontsize': 22,
              'figure.figsize': (12, 8),
              'axes.labelsize': 16,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.titlesize': 22}
    plt.rcParams.update(params)
    if new_params is not None:
        plt.rcParams.update(new_params)
    return plt


def update_ax_font(ax, size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)
    return ax
