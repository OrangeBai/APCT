import os
import pandas as pd
import torch
from settings.test_setting import TestParser
from core.tester import BaseTester, DualTester, restore_runs
from numpy.linalg import norm
from torch.nn.functional import one_hot, cosine_similarity
from core.dataloader import set_dataset
from core.utils import accuracy
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def cal_angle_dist(k, val, ground_truth):
    arrays = val['array']
    adv_std_diff = arrays[3] - arrays[1]
    fix_diff = arrays[2] - arrays[0]
    flt_diff = adv_std_diff - fix_diff
    sim_fix = cosine_similarity(torch.tensor(fix_diff).float(), ground_truth).numpy()
    sim_flt = cosine_similarity(torch.tensor(flt_diff).float(), ground_truth).numpy()
    ['fixed', ] * len(fix_diff) + ['float', ] * len(flt_diff)
    return pd.DataFrame({'value': np.concatenate([sim_fix, sim_flt]),
                         'path_type': ['fixed', ] * len(fix_diff) + ['float', ] * len(flt_diff),
                         'name': [k, ] * len(fix_diff) + [k, ] * len(flt_diff)})


if __name__ == '__main__':
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'adv_compare']
    load_args = TestParser(load_argsv).get_args()
    run_dirs = restore_runs(load_args)

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'test',
             '--test_mode', 'adv', '--attack', 'fgsm', '--eps', '0.0313', '--batch_size', '128']
    args = TestParser(argsv).get_args()
    dual_tester = DualTester(run_dirs, args)
    dual_tester.test(restart=False)

    _, test_set = set_dataset(args)
    ground = one_hot(torch.tensor(test_set.targets)).float()
    keys = ['std', 'noise_012', 'noise_025', 'fgsm_04', 'fgsm_08', 'pgd_04', 'pgd_08']

    data = pd.concat([cal_angle_dist(k, dual_tester.results[k], ground) for k in keys])

    large = 22
    med = 16
    small = 12
    params = {'axes.titlesize': med,
              'legend.fontsize': large,
              'figure.figsize': (12, 8),
              'axes.labelsize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.style.use('ggplot')
    plt.rcParams.update(params)
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.violinplot(data=data, x="name", y="value", hue="path_type", bw=0.2,
                   split=True, cut=0, inner=None, linewidth=0.25, ax=ax, palette=sns.color_palette("tab10"))
    ax.legend(fontsize=16)
    ax.set_xticklabels(
        ['Standard', 'Noise: 0.125', 'Noise: 0.25', 'FGSM: 4/255', 'FGSM: 8/255', 'PGD: 4/255', 'PGD: 8/255'])
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Cosine Similarity')
    save_path = os.path.join('plot', 'figs', 'direction.png')
    plt.savefig(save_path)

    # for k in keys:
    #     val = dual_tester.results[k]
    #     arrays = val['array']
    #     adv_std_diff = arrays[3] - arrays[1]
    #     fix_diff = arrays[2] - arrays[0]
    #     float_diff = adv_std_diff - fix_diff
    #
    #     res.extend(np.concatenate([cosine_similarity(torch.tensor(fix_diff).float(), ground).numpy(),
    #                           cosine_similarity(torch.tensor(float_diff).float(), ground).numpy()]))
    #     path_type.extend(['fixed', ] * len(fix_diff) + ['float', ] * len(float_diff))
    #     name.extend([k,] * len(fix_diff) + [k,] * len(float_diff))
    # data.extend([
    #     cosine_similarity(torch.tensor(fix_diff).float(), ground).numpy(),
    #     cosine_similarity(torch.tensor(float_diff).float(), ground).numpy()
    # ])
    # index.extend([k])

    # temp = {'fix': cosine_similarity(torch.tensor(fix_diff).float(), ground),
    #         'flt': cosine_similarity(torch.tensor(float_diff).float(), ground)}
    # df_dict[(k, 'fix')] = cosine_similarity(torch.tensor(fix_diff).float(), ground)
    # print("fix_diff similarity: {0}".format(cosine_similarity(torch.tensor(fix_diff).float(), ground)))
    # print("fix_diff similarity: {0}".format(cosine_similarity(torch.tensor(float_diff).float(), ground)))
    # std_fix_to_std = norm(arrays[3] - arrays[2], axis=-1) / norm(arrays[:, 0], axis=-1)
    # print("{0}:\nadv_std_diff:{1}\nfix_diff:{2}\nfloat_diff{3}".format(key, adv_std_diff.mean(), fix_diff.mean(), float_diff.mean()))
    #     print(1)
    # df = pd.DataFrame(data, index=index)
    # df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index, names=['first', 'second']))
    # df = sns.load_dataset("titanic")
    # sns.violinplot(x=df["age"])
#
#     _, test_set = set_dataset(args)
#     #
#     base_tester = BaseTester(run_dirs, args)
#     base_tester.test(restart=False)
#     print(1)
#
# import os
#
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
#
# sns.set()
# sns.set_theme(style="darkgrid")
# from config import *
# import pandas as pd
#
# large = 22
# med = 16
# small = 12
# params = {'axes.titlesize': large,
#           'legend.fontsize': large,
#           'figure.figsize': (12, 8),
#           'axes.labelsize': large,
#           'xtick.labelsize': large,
#           'ytick.labelsize': large,
#
#           'figure.titlesize': large}
# plt.rcParams.update(params)
# fig, ax = plt.subplots()
# np_file_0 = os.path.join(MODEL_PATH, 'exp', 'float_ratio_vgg_sigma_0.1.npy')
# np_file_1 = os.path.join(MODEL_PATH, 'exp', 'float_ratio_vgg_sigma_0.25.npy')
# flt0 = np.load(np_file_0)
# flt1 = np.load(np_file_1)
#
# d = flt0.mean(axis=2).T
# df = pd.DataFrame(data=d, columns=['STD', r'$\sigma$-0.05', r'$\sigma$-0.10', r'$\sigma$-0.25'])
# sns.violinplot(data=df, split=True, ax=ax)
