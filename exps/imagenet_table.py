import itertools
import os
import pandas as pd
import torch
import wandb
from core.scrfp import ApproximateAccuracy
from settings.test_setting import TestParser
from argparse import Namespace
from core.tester import SmoothedTester, restore_runs
from exps.plt_base import update_params, update_ax_font
from numpy.linalg import norm
from torch.nn.functional import one_hot, cosine_similarity
from core.dataloader import set_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy


def _update_res(res, base_path, base_name):
    line_space = np.linspace(0, 2.98, 13)
    names = {'smooth.txt': base_name,
             'scrfp-0.01.txt': base_name + ' + SCRFP2(0, 0.01)',
             'scrfp-0.05.txt': base_name + ' + SCRFP2(0, 0.05)',
             'scrfp-0.1.txt': base_name + ' + SCRFP2(0, 0.10)'}
    res.update({v: ApproximateAccuracy(os.path.join(base_path, k)).at_radii(line_space) for k, v in names.items()})
    return res


def _print_acr(res, i, base_path):
    names = {'smooth.txt': 'Benchmark',
             'scrfp-0.01.txt': 'SCRFP-2(0, 0.01)',
             'scrfp-0.05.txt': 'SCRFP-2(0, 0.05)',
             'scrfp-0.1.txt': 'SCRFP-2(0, 0.10)'}
    acr = {v: ApproximateAccuracy(os.path.join(base_path, k)).acr() for k, v in names.items()}
    res[i] = acr
    return


if __name__ == '__main__':
    table_dict = {}
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\benchmark\noise_0.25", 'Smooth Classifer')
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_0.25", 'SmoothAdv')
    df = 100 * pd.DataFrame(table_dict).T
    df = df[list(range(1, 13)) + [0, 13]]
    df.columns = (list(np.linspace(0.25, 3, 12)) + ['Clean', 'ACR'])
    print(df.to_latex(float_format="%.1f"))

    table_dict = {}
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\benchmark\noise_0.50", 'Smooth Classifer')
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_0.50", 'SmoothAdv')
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\mix\noise_0.50", 'SmoothMix')
    df = 100 * pd.DataFrame(table_dict).T
    df = df[list(range(1, 13)) + [0, 13]]
    df.columns = (list(np.linspace(0.25, 3, 12)) + ['Clean', 'ACR'])
    print(df.to_latex(float_format="%.1f"))

    table_dict = {}
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\benchmark\noise_1.00", 'Smooth Classifer')
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_1.00", 'SmoothAdv')
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\mix\noise_1.00", 'SmoothMix')
    df = 100 * pd.DataFrame(table_dict).T
    df = df[list(range(1, 13)) + [0, 13]]
    df.columns = (list(np.linspace(0.25, 3, 12)) + ['Clean', 'ACR'])
    print(df.to_latex(float_format="%.1f"))

    acr_dict = {}
    _print_acr(acr_dict, 0, r"E:\Experiments\SCRFP\imagenet\benchmark\noise_0.25")
    _print_acr(acr_dict, 1, r"E:\Experiments\SCRFP\imagenet\benchmark\noise_0.50")
    _print_acr(acr_dict, 2, r"E:\Experiments\SCRFP\imagenet\benchmark\noise_1.00")

    _print_acr(acr_dict, 3, r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_0.25")
    _print_acr(acr_dict, 4, r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_0.50")
    _print_acr(acr_dict, 5, r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_1.00")

    _print_acr(acr_dict, 6, r"E:\Experiments\SCRFP\imagenet\mix\noise_0.50")
    _print_acr(acr_dict, 7, r"E:\Experiments\SCRFP\imagenet\mix\noise_1.00")
    df = pd.DataFrame(acr_dict)
    print(df.to_latex(float_format='%.3f'))
    print(1)