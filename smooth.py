import itertools
import os
import pandas as pd
import torch
import wandb
from core.scrfp import ApproximateAccuracy
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
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16']
    load_args = TestParser(load_argsv).get_args()
    runs = restore_runs(load_args)

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--test_mode', 'smoothed_certify',
             '--smooth_model', 'smooth']
    args = TestParser(argsv).get_args()

    test_names = [i.format(args.sigma) for i in ['flt_{}_0.05', 'flt_{}_0.10', 'flt_{}_0.15', 'flt_{}_0.20', 'std_{}']]
    run_dirs = {run: run_dir for run, run_dir in runs.items() if run.name in test_names}

    tester = SmoothedTester(run_dirs, args)
    res1 = tester.test(restart=True)

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'dual', '--test_mode', 'smoothed_certify',
             '--smooth_model', 'SCRFP', '--eta_float', '-0.1']
    args = TestParser(argsv).get_args()
    tester = SmoothedTester(run_dirs, args)
    res2 = tester.test(restart=True)




    # smooth, scrfp = {}, {}
    # for n, p in run_dirs.items():
    #     smooth_path = os.path.join(p, 'test', 'smooth.txt')
    #     scrfp_path = os.path.join(p, 'test', 'scrfp.txt')
    #     smooth[n] = ApproximateAccuracy(smooth_path).at_radii(np.linspace(0, 2, 400))
    #     scrfp[n] = ApproximateAccuracy(scrfp_path).at_radii(np.linspace(0, 2, 400))
    # print(1)