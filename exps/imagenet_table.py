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


if __name__ == '__main__':
    table_dict = {}
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\benchmark\noise_0.25", 'Smooth Classifer')
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_0.25", 'SmoothAdv')
    df = 100 * pd.DataFrame(table_dict).T
    df = df[list(range(1, 13)) + [0]]
    df.columns = (list(np.linspace(0.25, 3, 12)) + ['Clean'])
    print(df.to_latex(float_format="%.1f"))

    table_dict = {}
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\benchmark\noise_0.50", 'Smooth Classifer')
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_0.50", 'SmoothAdv')
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\mix\noise_0.50", 'SmoothMix')
    df = 100 * pd.DataFrame(table_dict).T
    df = df[list(range(1, 13)) + [0]]
    df.columns = (list(np.linspace(0.25, 3, 12)) + ['Clean'])
    print(df.to_latex(float_format="%.1f"))

    table_dict = {}
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\benchmark\noise_1.00", 'Smooth Classifer')
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_1.00", 'SmoothAdv')
    _update_res(table_dict, r"E:\Experiments\SCRFP\imagenet\mix\noise_1.00", 'SmoothMix')
    df = 100 * pd.DataFrame(table_dict).T
    df = df[list(range(1, 13)) + [0]]
    df.columns = (list(np.linspace(0.25, 3, 12)) + ['Clean'])
    print(df.to_latex(float_format="%.1f"))
