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

    # Plot 0.125
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--test_mode', 'smoothed_certify',
             '--smooth_model', 'smooth', '--sigma', '0.125']
    args = TestParser(argsv).get_args()
    test_names = [i.format(args.sigma) for i in ['flt_{}_0.01', 'flt_{}_0.02', 'std_{}']]
    run_dirs = {run: run_dir for run, run_dir in runs.items() if run.name in test_names}
    smooth, scrfp = {}, {}
    for n, p in run_dirs.items():
        smooth_path = os.path.join(p, 'test', 'smooth.txt')
        scrfp_path = os.path.join(p, 'test', 'scrfp.txt')
        smooth[n.name] = ApproximateAccuracy(smooth_path).at_radii(np.linspace(0, 0.5, 20))
        scrfp[n.name] = ApproximateAccuracy(scrfp_path).at_radii(np.linspace(0, 0.5, 20))

    plt = update_params(plt, {'legend.fontsize': 12})
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(np.linspace(0, 0.5, 20), smooth['std_0.125'], label='STD', linestyle='-', color='r')
    ax.plot(np.linspace(0, 0.5, 20), smooth['flt_0.125_0.01'], label='SCRFP-2(0.01, 0)', linestyle='-', color='g')
    ax.plot(np.linspace(0, 0.5, 20), smooth['flt_0.125_0.02'], label='SCRFP-2(0.02, 0)', linestyle='-', color='b')
    
    ax.plot(np.linspace(0, 0.5, 20), scrfp['std_0.125'], label='SCRFP-2(0, -0.1)', linestyle='-.', color='r')
    ax.plot(np.linspace(0, 0.5, 20), scrfp['flt_0.125_0.01'], label='SCRFP-2(0.01, -0.1)', linestyle='-.', color='g')
    ax.plot(np.linspace(0, 0.5, 20), scrfp['flt_0.125_0.02'], label='SCRFP-2(0.02, -0.1)', linestyle='-.', color='b')
    
    ax.legend()
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0.55, 0.85])
    plt.show()
    print(1)

    # Plot 0.25
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--test_mode', 'smoothed_certify',
             '--smooth_model', 'smooth', '--sigma', '0.25']
    args = TestParser(argsv).get_args()
    test_names = [i.format(args.sigma) for i in ['flt_{}_0.01', 'flt_{}_0.02', 'std_{}']]
    run_dirs = {run: run_dir for run, run_dir in runs.items() if run.name in test_names}
    smooth, scrfp = {}, {}
    for n, p in run_dirs.items():
        smooth_path = os.path.join(p, 'test', 'smooth.txt')
        scrfp_path = os.path.join(p, 'test', 'scrfp.txt')
        smooth[n.name] = ApproximateAccuracy(smooth_path).at_radii(np.linspace(0, 1, 40))
        scrfp[n.name] = ApproximateAccuracy(scrfp_path).at_radii(np.linspace(0, 1, 40))

    plt = update_params(plt, {'legend.fontsize': 12})
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(np.linspace(0, 1, 40), smooth['std_0.25'], label='STD', linestyle='-', color='r')
    ax.plot(np.linspace(0, 1, 40), smooth['flt_0.25_0.01'], label='SCRFP-2(0.01, 0)', linestyle='-', color='g')
    ax.plot(np.linspace(0, 1, 40), smooth['flt_0.25_0.02'], label='SCRFP-2(0.02, 0)', linestyle='-', color='b')

    ax.plot(np.linspace(0, 1, 40), scrfp['std_0.25'], label='SCRFP-2(0, -0.1)', linestyle='-.', color='r')
    ax.plot(np.linspace(0, 1, 40), scrfp['flt_0.25_0.01'], label='SCRFP-2(0.01, -0.1)', linestyle='-.', color='g')
    ax.plot(np.linspace(0, 1, 40), scrfp['flt_0.25_0.02'], label='SCRFP-2(0.02, -0.1)', linestyle='-.', color='b')

    ax.legend()
    ax.set_ylim([0.35, 0.8])
    ax.set_xlim([0, 0.9])
    plt.show()

    # Plot 0.5
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--test_mode', 'smoothed_certify',
             '--smooth_model', 'smooth', '--sigma', '0.5']
    args = TestParser(argsv).get_args()
    test_names = [i.format(args.sigma) for i in ['flt_{}_0.01', 'flt_{}_0.02', 'std_{}']]
    run_dirs = {run: run_dir for run, run_dir in runs.items() if run.name in test_names}
    smooth, scrfp = {}, {}
    for n, p in run_dirs.items():
        smooth_path = os.path.join(p, 'test', 'smooth.txt')
        scrfp_path = os.path.join(p, 'test', 'scrfp.txt')
        smooth[n.name] = ApproximateAccuracy(smooth_path).at_radii(np.linspace(0, 1.5, 40))
        scrfp[n.name] = ApproximateAccuracy(scrfp_path).at_radii(np.linspace(0, 1.5, 40))

    plt = update_params(plt, {'legend.fontsize': 12})
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(np.linspace(0, 1.5, 40), smooth['std_0.5'], label='STD', linestyle='-', color='r')
    ax.plot(np.linspace(0, 1.5, 40), smooth['flt_0.5_0.01'], label='SCRFP-2(0.01, 0)', linestyle='-', color='g')
    ax.plot(np.linspace(0, 1.5, 40), smooth['flt_0.5_0.02'], label='SCRFP-2(0.02, 0)', linestyle='-', color='b')

    ax.plot(np.linspace(0, 1.5, 40), scrfp['std_0.5'], label='SCRFP-2(0, -0.1)', linestyle='-.', color='r')
    ax.plot(np.linspace(0, 1.5, 40), scrfp['flt_0.5_0.01'], label='SCRFP-2(0.01, -0.1)', linestyle='-.', color='g')
    ax.plot(np.linspace(0, 1.5, 40), scrfp['flt_0.5_0.02'], label='SCRFP-2(0.02, -0.1)', linestyle='-.', color='b')

    ax.legend()
    ax.set_ylim([0.35, 0.8])
    ax.set_xlim([0, 1.4])
    plt.show()