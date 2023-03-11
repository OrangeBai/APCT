import os
import pandas as pd
import torch
from settings.test_setting import TestParser
from core.tester import BaseTester, DualTester, AveragedFloatTester, FltRatioTester, LipDistTester, restore_runs, LipNormTester
from exps.plot.plt_base import update_params, update_ax_font
from numpy.linalg import norm
from torch.nn.functional import one_hot, cosine_similarity
from core.dataloader import set_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy


if __name__ == '__main__':
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'adv_compare']
    load_args = TestParser(load_argsv).get_args()
    run_dirs = restore_runs(load_args)

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'test',
             '--test_mode', 'adv', '--attack', 'noise', '--sigma', '0.25', '--batch_size', '128']
    args = TestParser(argsv).get_args()
    # dual_tester = DualTester(run_dirs, args)
    # results = dual_tester.test(restart=False)
    # TODO check how to set iter dict according to given list (ordered dict)

    keys = ['std', 'noise_012', 'noise_025', 'fgsm_04', 'fgsm_08', 'pgd_04', 'pgd_08']
    legend_keys = ['Standard', 'Noise: 0.125', 'Noise: 0.25', 'FGSM: 4/255', 'FGSM: 8/255', 'PGD: 4/255', 'PGD: 8/255']
    ############################    Compute and plot the direction   #####################################
    # data = dual_tester.format_angel_result(keys, legend_keys)
    #
    # # plot the result
    # plt = update_params(plt)
    # fig, ax = plt.subplots(figsize=(16, 9))
    # sns.violinplot(data=data, x="name", y="value", hue="path_type", bw=0.2,
    #                split=True, cut=0, inner=None, linewidth=0.25, ax=ax, palette=sns.color_palette("tab10"))
    # ax.legend(fontsize=16)
    # ax.set_xticklabels(legend_keys)
    # ax.set_xlabel('Experiment')
    # ax.set_ylabel('Cosine Similarity')
    # save_path = os.path.join('figs', 'pgd_direction.png')
    # # plt.show()
    # plt.savefig(save_path)
    # ############################    Compute and plot the distance   ######################################
    # data = dual_tester.format_dist_result(keys, legend_keys)

    # plot the result
    # plt = update_params(plt)
    # _, ax = plt.subplots(figsize=(16, 9))
    # for k, legend_k in zip(keys, legend_keys):
    #     ax.scatter(x='fix', y='flt', data=data[legend_k], label=legend_k, s=1)
    # ax.legend(markerscale=10, fontsize=16)
    # ax.set_xlim([0, 1.5])
    # ax.set_ylim([0, 18])
    # ax.tick_params(axis='x', labelsize=20)
    # ax.set_xlabel('Fixed Path Distance', fontsize=22)
    # ax.set_ylabel('Float Path Distance', fontsize=22)
    # save_path = os.path.join('figs', 'pgd_distance.png')
    # # plt.show()
    # plt.savefig(save_path)
    #
    ##########################  Averaged Float Teste    #####################
    # exps = ['std', 'noise_025', 'noise_012']
    # tester = AveragedFloatTester({k: run_dirs[k] for k in exps}, args)
    # tester.test(restart=True)
    # print(1)
    # res = tester.compute_norm()
    # print(pd.DataFrame(res).T.to_latex())

    ###########################     Ratio of fixed and float vulnerable ##################
    # table = {}
    # data = dual_tester.float_fixed_vulnerable(keys, legend_keys)
    # for k, legend_k in zip(keys, legend_keys):
    #     v = dual_tester.results.get(k)
    #     t2 = {}
    #     for k2 in ['top1', 'top1_adv', 'top1_fix_std', 'top1_fix_adv']:
    #         t2[k2] = v.get(k2)
    #     table[legend_k] = t2
    # for k, v in data.items():
    #     table[k]['Float Vulnerable'] = v
    # print(1)
    ##########################      Lip Norm #########################
    lip_norm_tester = LipNormTester(run_dirs, args)
    data = lip_norm_tester.test(restart=False)
    label = one_hot(torch.tensor(set_dataset(args)[1].targets)).numpy()
    # print(1)
    # for k, legend_keys in zip(keys, legend_keys):
    #     v = np.concatenate(data[k], axis=1)
    #     fixed = (norm(v[0] * label, axis=1) > norm(v[1] * label, axis=1)).sum()



    ############################    Compute and plot the distance   ######################################
    # flt_ratio_tester = FltRatioTester(run_dirs, args)
    # data = flt_ratio_tester.test(restart=False)
    #
    # plt = update_params(plt)
    # fig, ax = plt.subplots(figsize=(20, 8))
    #
    # width = 0.12
    #
    # for i, (k, legend_k) in enumerate(zip(keys, legend_keys)):
    #     ratio = np.array(data[k]).mean(axis=0)
    #     loc = np.arange(len(ratio)) + (i - 3) * width
    #     ax.bar(loc, ratio, width=width, label=legend_k)
    #
    # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    #     item.set_fontsize(20)
    # fig.legend(loc='upper right', bbox_to_anchor=(0.80, 0.88), prop={'size': 12})
    # save_path = os.path.join('figs', 'flt_num.png')
    # # plt.show()
    # plt.savefig(save_path, bbox_inches='tight')

    ############################    Compute and plot the distance   ######################################
    # lip_dist_tester = LipDistTester(run_dirs, args)
    # data = lip_dist_tester.test(restart=False)
    # lip_mean, lip_var = lip_dist_tester.get_mean_var('lip')
    # dist_mean, dist_var = lip_dist_tester.get_mean_var('dist')
    #
    # plt = update_params(plt)
    # fig, ax = plt.subplots(figsize=(16, 9))
    # for i, (k, legend_k) in enumerate(zip(keys, legend_keys)):
    #     ax.scatter(np.log(lip_mean[k]), np.log(dist_mean[k]), label=legend_k, s=1)
    # update_ax_font(ax, 18)
    # fig.legend(loc='lower right', bbox_to_anchor=(0.90, 0.1), markerscale=10)
    # ax.set_xlabel('log(Mean of Lipschitz Constant)', fontsize=18)
    # ax.set_ylabel('log(Mean of Prediction Distance)', fontsize=18)
    # ax.set_xlim(-1, 6)
    # ax.set_ylim(-2, 3)
    # # plt.show()
    # plt.savefig('figs/lip_mean_dist_mean', bbox_inches='tight')
    #
    # plt = update_params(plt)
    # _, ax = plt.subplots(figsize=(16, 9))
    # for i, (k, legend_k) in enumerate(zip(keys, legend_keys)):
    #     ax.scatter(np.log(lip_mean[k]), np.log(lip_var[k]), label=legend_k, s=1)
    # update_ax_font(ax, 18)
    # ax.legend(loc='upper left', prop={'size': 18}, markerscale=10)
    # ax.set_xlabel('log(Mean of Lipschitz Constant)', fontsize=18)
    # ax.set_ylabel('log(Variance of Lipschitz Constant)', fontsize=18)
    # ax.set_xlim(-1, 5)
    # ax.set_ylim(-8, 4)
    # # plt.show()
    # plt.savefig('figs/lip_mean_lip_var', bbox_inches='tight')
    #
    # plt = update_params(plt)
    # fig, ax = plt.subplots(figsize=(16, 9))
    # for i, (k, legend_k) in enumerate(zip(keys, legend_keys)):
    #     ax.scatter(np.log(lip_mean[k]), np.log(dist_var[k]), label=legend_k, s=1)
    # ax.legend(loc='upper left', prop={'size': 18}, markerscale=10)
    # update_ax_font(ax, 18)
    # ax.set_xlabel('log(Mean of Lipschitz Constant)', fontsize=18)
    # ax.set_ylabel('log(Variance of Prediction Distance)', fontsize=18)
    # ax.set_xlim(-1, 5)
    # ax.set_ylim(-10, 5)
    # # plt.show()
    # plt.savefig('figs/lip_mean_dist_var', bbox_inches='tight')
