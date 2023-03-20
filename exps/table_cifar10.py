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


    res = {}
    for sigma in ['0.125', '0.25', '0.5']:
        argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--test_mode', 'smoothed_certify',
                 '--smooth_model', 'smooth', '--sigma', sigma]
        args = TestParser(argsv).get_args()
        test_names = [i.format(args.sigma) for i in ['flt_{}_0.01', 'flt_{}_0.02', 'std_{}']]
        run_dirs = {run: run_dir for run, run_dir in runs.items() if run.name in test_names}
        for n, p in run_dirs.items():
            smooth_path = os.path.join(p, 'test', 'smooth.txt')
            scrfp_path = os.path.join(p, 'test', 'scrfp.txt')
            res['smooth_' + n.name] = ApproximateAccuracy(smooth_path).at_radii(np.linspace(0, 2, 9))
            res['scrfp_' + n.name] = ApproximateAccuracy(scrfp_path).at_radii(np.linspace(0, 2, 9))

    res = {k: res[k] for k in sorted(res)}
    res = pd.DataFrame(res).T
    res.to_latex()
    print(1)