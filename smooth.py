import itertools
import os
import pandas as pd
import torch
import wandb

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
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'dual_smooth']
    load_args = TestParser(load_argsv).get_args()
    run_dirs = restore_runs(load_args)

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'dual', '--test_mode', 'smoothed_certify',
             '--smooth_model', 'smooth']
    args = TestParser(argsv).get_args()

    test_names = [i.format(args.sigma) for i in ['flt_{}_0.05', 'fix_{}_-0.05', 'std_{}']]
    run_dirs = {name: run_dirs[name] for name in test_names}

    tester = SmoothedTester(run_dirs, args)
    res1 = tester.test(restart=False)

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'dual', '--test_mode', 'smoothed_certify',
             '--smooth_model', 'SCRFP', '--eta_float', '-0.1']
    args = TestParser(argsv).get_args()
    tester = SmoothedTester(run_dirs, args)
    res2 = tester.test(restart=False)
    print(1)
    wandb.init()
    api = wandb.Api()
    a = api.runs(args.project)

    wandb.init(project="preemptable", resume=True)