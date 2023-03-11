import os
import pandas as pd
import torch
from settings.test_setting import TestParser
from argparse import Namespace
from core.tester import SmoothedTester, restore_runs
from exps.plot.plt_base import update_params, update_ax_font
from numpy.linalg import norm
from torch.nn.functional import one_hot, cosine_similarity
from core.dataloader import set_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy

if __name__ == '__main__':
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'dual']
    load_args = TestParser(load_argsv).get_args()
    run_dirs = restore_runs(load_args)

    test_names = ['float_0.00', 'float_0.1', 'float_0.2',
                  'fixed_0.00', 'fixed_-0.05', 'fixed_-0.1']
    run_dirs = {name: run_dirs[name] for name in test_names}
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'dual', '--test_mode', 'smoothed_certify',
             '--smooth_model', 'smooth']
    args = TestParser(argsv).get_args()

    tester = SmoothedTester(run_dirs, args)
    res1 = tester.test(restart=False)

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'dual', '--test_mode', 'smoothed_certify',
             '--smooth_model', 'SCRFP', '--eta_float', '-0.1']
    args = TestParser(argsv).get_args()
    tester = SmoothedTester(run_dirs, args)
    res2 = tester.test(restart=True)
    print(1)
