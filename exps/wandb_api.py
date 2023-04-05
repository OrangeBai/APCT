import matplotlib.pyplot as plt
import wandb
import seaborn as sns
from settings.test_setting import TestParser

if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'express_batch']
    args = TestParser(argsv).get_args()
    # os.environ["WANDB_DIR"] = args.model_dir
    api = wandb.Api(timeout=60)
    runs = api.runs(args.project)

    sns.set()
    sns.set_theme(style="darkgrid")
    large = 32
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
    # Project is specified by <entity/project-name>
    numeric_const_pattern = r"""
         [-+]? # optional sign
         (?:
             (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
             |
             (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
         )
         # followed by optional exponent part if desired
         (?: [Ee] [+-]? \d+ ) ?
         """

    names = {'batch_32': 'batch size: 32', 'batch_64': 'batch size: 64', 'batch_132': 'batch size:132',
             'batch_256': 'batch size:256','batch_512': 'batch size:512'}
    all_float = {}

    lines = []
    for i in [1, 6,  11]:
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62732', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd32',
                 '#17becf']
        for c, cur_run in zip(color, runs):
            if cur_run.name not in names.keys():
                continue
            entropy_name = 'val_set/entropy_mean/layer_{:02}'.format(i)
            var_name = 'val_set/entropy_var/layer_{:02}'.format(i)
            data = cur_run.history(keys=[entropy_name])
            x = data['_step'] * 400
            y = data[entropy_name]
            ax.plot(x, y, color=c, label=names[cur_run.name])
        ax.set_xlabel('Steps', fontsize=32)
        ax.set_ylabel('Mean of Neuron Entropy', fontsize=32)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(32)
        plt.savefig('figs/mean_layer_{0:02d}.png'.format(i), bbox_inches='tight')

    lines = []
    for i in [1, 6,  11]:
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62732', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd32',
                 '#17becf']
        for c, cur_run in zip(color, runs):
            if cur_run.name not in names.keys():
                continue
            entropy_name = 'val_set/entropy_mean/layer_{:02}'.format(i)
            var_name = 'val_set/entropy_var/layer_{:02}'.format(i)
            data = cur_run.history(keys=[var_name])
            x = data['_step'] * 400
            y = data[var_name]
            ax.plot(x, y, color=c, label=names[cur_run.name])
        ax.set_xlabel('Steps', fontsize=32)
        ax.set_ylabel('Variance of Neuron Entropy', fontsize=32)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(32)
        plt.savefig('figs/var_layer_{0:02d}.png'.format(i), bbox_inches='tight')

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62732', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd32',
             '#17becf']
    for c, cur_run in zip(color, runs):
        if cur_run.name not in names.keys():
            continue
        entropy_name = 'train/top1'.format(i)
        data = cur_run.history(keys=[entropy_name])
        x = data['_step'] * 400
        y = data[entropy_name]
        ax.plot(x, y, color=c, label=names[cur_run.name])
    ax.set_xlabel('Steps', fontsize=32)
    ax.set_ylabel('Train Accuracy', fontsize=32)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(32)
    plt.legend()
    plt.savefig('figs/train_acc.png'.format(i), bbox_inches='tight')

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62732', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd32',
             '#17becf']
    for c, cur_run in zip(color, runs):
        if cur_run.name not in names.keys():
            continue
        entropy_name = 'val/top1'.format(i)
        data = cur_run.history(keys=[entropy_name])
        x = data['_step'] * 400
        y = data[entropy_name]
        ax.plot(x, y, color=c, label=names[cur_run.name])
    ax.set_xlabel('Steps', fontsize=32)
    ax.set_ylabel('Validation Accuracy', fontsize=32)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(32)
    plt.legend()
    plt.savefig('figs/val_acc.png'.format(i), bbox_inches='tight')
    print(1)