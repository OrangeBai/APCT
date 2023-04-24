import pandas as pd
from settings.test_setting import TestParser
from argparse import Namespace
from core.tester import RobustnessTester, restore_runs
from exps import update_params, update_ax_font
import matplotlib.pyplot as plt

if __name__ == '__main__':
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'dual']
    load_args = TestParser(load_argsv).get_args()
    run_dirs = restore_runs(load_args)

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'dual']
    args = TestParser(argsv).get_args()
    namespaces = {
        'noise': Namespace(dataset=args.dataset, attack='noise', sigma=0.125),
        'fgsm_04': Namespace(dataset=args.dataset, attack='fgsm', eps=4 / 255),
        'fgsm_08': Namespace(dataset=args.dataset, attack='fgsm', eps=8 / 255),
        'pgd_04': Namespace(dataset=args.dataset, attack='pgd', eps=4 / 255),
        'pgd_08': Namespace(dataset=args.dataset, attack='pgd', eps=8 / 255)
    }

    tester = RobustnessTester(run_dirs, args, namespaces)
    res = tester.test(restart=False)

    df_res = pd.DataFrame(res)

    float_columns = [i for i in df_res.columns if 'float' in i]
    float_res = df_res[float_columns]
    float_res.columns = [float(name.split('_')[1]) for name in float_columns if 'float' in name]
    float_res = float_res.reindex(sorted(float_res.columns), axis=1)

    plt = update_params(plt, {'legend.fontsize': 12})
    fig, ax = plt.subplots(figsize=(16, 10), nrows=3, ncols=2)
    ax[0][0].plot(float_res.loc['top1'], label='Clean (Top-1)')
    ax[0][0].plot(float_res.loc['top1'][0], markersize=10, marker='*', color='b')

    ax[0][1].plot(float_res.loc['pgd_04_top1'], label='PGD:  4/255 (Top-1)')
    ax[0][1].plot(float_res.loc['pgd_04_top1'][0], markersize=10, marker='*', color='b')

    ax[1][0].plot(float_res.loc['fgsm_04_top1'], label='FGSM: 4/255 (Top-1)')
    ax[1][0].plot(float_res.loc['fgsm_04_top1'][0], markersize=10, marker='*', color='b')

    ax[1][1].plot(float_res.loc['fgsm_04_top5'], label='FGSM: 8/255 (Top-5)')
    ax[1][1].plot(float_res.loc['fgsm_04_top5'][0], markersize=10, marker='*', color='b')

    ax[1][1].plot(float_res.loc['fgsm_08_top5'][0], markersize=10, marker='*', color='b')
    ax[1][1].plot(float_res.loc['fgsm_08_top5'][0], markersize=10, marker='*', color='b')
    # ax[0].set_ylim([80, 90])
    # ax[1].legend()

    ax[2][0].plot(float_res.loc['fgsm_08_top1'], label='FGSM: 8/255 (Top-1)')
    ax[2][0].plot(float_res.loc['fgsm_08_top1'][0], markersize=10, marker='*', color='b')

    ax[2][1].plot(float_res.loc['fgsm_08_top5'], label='FGSM: 4/255 (Top-5)')
    ax[2][1].plot(float_res.loc['fgsm_08_top5'][0], markersize=10, marker='*', color='b')

    fig.canvas.draw()
    for row, x in enumerate(ax):
        for y in x:
            y.legend()
            labels = [item.get_text() for item in y.get_xticklabels()]
            labels[3] = 'benchmark'
            y.get_xticklabels()[3].set_c('b')
            y.set_xticklabels(labels)
            if row == 2:
                y.set_xlabel('$\eta$')
            y.set_ylabel('accuracy')
    plt.savefig('figs/float_res', bbox_inches='tight')


    fixed_columns = [i for i in df_res.columns if 'fixed' in i]
    fixed_res = df_res[fixed_columns]
    fixed_res.columns = [float(name.split('_')[1]) for name in fixed_columns if 'fixed' in name]
    fixed_res = fixed_res.reindex(sorted(fixed_res.columns), axis=1)
    fixed_res[0] = float_res[0]

    plt = update_params(plt, {'legend.fontsize': 12})
    fig, ax = plt.subplots(figsize=(16, 10), nrows=3, ncols=2)
    ax[0][0].plot(fixed_res.loc['top1'], label='Clean (Top-1)')
    ax[0][0].plot(fixed_res.loc['top1'][0], markersize=10, marker='*', color='b')

    ax[0][1].plot(fixed_res.loc['pgd_04_top1'], label='PGD:  4/255 (Top-1)')
    ax[0][1].plot(fixed_res.loc['pgd_04_top1'][0], markersize=10, marker='*', color='b')

    ax[1][0].plot(fixed_res.loc['fgsm_04_top1'], label='FGSM: 4/255 (Top-1)')
    ax[1][0].plot(fixed_res.loc['fgsm_04_top1'][0], markersize=10, marker='*', color='b')

    ax[1][1].plot(fixed_res.loc['fgsm_04_top5'], label='FGSM: 8/255 (Top-5)')
    ax[1][1].plot(fixed_res.loc['fgsm_04_top5'][0], markersize=10, marker='*', color='b')

    ax[1][1].plot(fixed_res.loc['fgsm_08_top5'][0], markersize=10, marker='*', color='b')
    ax[1][1].plot(fixed_res.loc['fgsm_08_top5'][0], markersize=10, marker='*', color='b')
    # ax[0].set_ylim([80, 90])
    # ax[1].legend()

    ax[2][0].plot(fixed_res.loc['fgsm_08_top1'], label='FGSM: 8/255 (Top-1)')
    ax[2][0].plot(fixed_res.loc['fgsm_08_top1'][0], markersize=10, marker='*', color='b')

    ax[2][1].plot(fixed_res.loc['fgsm_08_top5'], label='FGSM: 4/255 (Top-5)')
    ax[2][1].plot(fixed_res.loc['fgsm_08_top5'][0], markersize=10, marker='*', color='b')

    fig.canvas.draw()
    for row, x in enumerate(ax):
        for y in x:
            y.legend()
            labels = [item.get_text() for item in y.get_xticklabels()]
            labels[3] = 'benchmark'
            y.get_xticklabels()[3].set_c('b')
            y.set_xticklabels(labels)
            if row == 2:
                y.set_xlabel('$\eta$')
            y.set_ylabel('accuracy')
    plt.savefig('figs/fixed_res', bbox_inches='tight')
    print(1)
