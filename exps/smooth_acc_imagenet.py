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

if __name__ == '__main__':
    # benchmark
    base_path = r"E:\Experiments\SCRFP\imagenet\benchmark\noise_0.25"
    names = ['smooth.txt', 'scrfp-0.01.txt', 'scrfp-0.05.txt', 'scrfp-0.1.txt']
    line_space = np.linspace(0, 0.8, 60)
    res = {name: ApproximateAccuracy(os.path.join(base_path, name)).at_radii(line_space) for name in names}
    plt = update_params(plt, {'legend.fontsize': 16, 'axes.labelsize': 22,
                              'xtick.labelsize': 22,
                              'ytick.labelsize': 22, })
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(line_space, res['smooth.txt'], label='Smoothed Classifier', linestyle='solid', color='r')
    ax.plot(line_space, res['scrfp-0.01.txt'], label='Smoothed Classifier + SCRFP2(0, 0.01)', linestyle='solid',
            linewidth=2, color='g')
    ax.plot(line_space, res['scrfp-0.05.txt'], label='Smoothed Classifier + SCRFP2(0, 0.05)', linestyle='solid',
            linewidth=2, color='b')
    ax.plot(line_space, res['scrfp-0.1.txt'], label='Smoothed Classifier + SCRFP2(0, 0.10)', linestyle='solid',
            linewidth=2, color='y')

    base_path = r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_0.25"
    names = ['smooth.txt', 'scrfp-0.01.txt', 'scrfp-0.05.txt', 'scrfp-0.1.txt']

    res = {name: ApproximateAccuracy(os.path.join(base_path, name)).at_radii(line_space) for name in names}
    ax.plot(line_space, res['smooth.txt'], label='SmoothAdv', linestyle='-', color='r')
    ax.plot(line_space, res['scrfp-0.01.txt'], label='SmoothAdv + SCRFP2(0, 0.01)', linestyle='dashed', linewidth=2,
            color='g')
    ax.plot(line_space, res['scrfp-0.05.txt'], label='SmoothAdv + SCRFP2(0, 0.05)', linestyle='dashed', linewidth=2,
            color='b')
    ax.plot(line_space, res['scrfp-0.1.txt'], label='SmoothAdv + SCRFP2(0, 0.10)', linestyle='dashed', linewidth=2,
            color='y')

    ax.legend()
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0.3, 0.7])
    ax.set_xlabel('Radius')
    ax.set_ylabel('Accuracy')
    # plt.show()
    plt.savefig('figs/imagenet_0.25.png', bbox_inches='tight')
    print(1)

    ######################################## Noise 0.50 ############################
    base_path = r"E:\Experiments\SCRFP\imagenet\benchmark\noise_0.50"
    names = ['smooth.txt', 'scrfp-0.01.txt', 'scrfp-0.05.txt', 'scrfp-0.1.txt']
    line_space = np.linspace(0, 1.6, 60)

    res = {name: ApproximateAccuracy(os.path.join(base_path, name)).at_radii(line_space) for name in names}
    plt = update_params(plt, {'legend.fontsize': 16, 'axes.labelsize': 22,
                              'xtick.labelsize': 22,
                              'ytick.labelsize': 22, })
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(line_space, res['smooth.txt'], label='Smoothed Classifier', linestyle='solid', linewidth=2, color='r')
    ax.plot(line_space, res['scrfp-0.01.txt'], label='Smoothed Classifier + SCRFP2(0, 0.01)', linestyle='solid',
            linewidth=2, color='g')
    ax.plot(line_space, res['scrfp-0.05.txt'], label='Smoothed Classifier + SCRFP2(0, 0.05)', linestyle='solid',
            linewidth=2, color='b')
    ax.plot(line_space, res['scrfp-0.1.txt'], label='Smoothed Classifier + SCRFP2(0, 0.10)', linestyle='solid',
            linewidth=2, color='y')

    base_path = r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_0.50"
    names = ['smooth.txt', 'scrfp-0.01.txt', 'scrfp-0.05.txt', 'scrfp-0.1.txt']

    res = {name: ApproximateAccuracy(os.path.join(base_path, name)).at_radii(line_space) for name in names}
    ax.plot(line_space, res['smooth.txt'], label='SmoothAdv', linestyle='dashed', color='r')
    ax.plot(line_space, res['scrfp-0.01.txt'], label='SmoothAdv + SCRFP2(0, 0.01)', linestyle='dashed', linewidth=2,
            color='g')
    ax.plot(line_space, res['scrfp-0.05.txt'], label='SmoothAdv + SCRFP2(0, 0.05)', linestyle='dashed', linewidth=2,
            color='b')
    ax.plot(line_space, res['scrfp-0.1.txt'], label='SmoothAdv + SCRFP2(0, 0.10)', linestyle='dashed', linewidth=2,
            color='y')

    base_path = r"E:\Experiments\SCRFP\imagenet\mix\noise_0.50"
    names = ['smooth.txt', 'scrfp-0.01.txt', 'scrfp-0.05.txt', 'scrfp-0.1.txt']

    res = {name: ApproximateAccuracy(os.path.join(base_path, name)).at_radii(line_space) for name in names}
    ax.plot(line_space, res['smooth.txt'], label='SmoothMix', linestyle='dotted', linewidth=2, color='r')
    ax.plot(line_space, res['scrfp-0.01.txt'], label='SmoothMix + SCRFP2(0, 0.01)', linestyle='dotted', linewidth=2,
            color='g')
    ax.plot(line_space, res['scrfp-0.05.txt'], label='SmoothMix + SCRFP-2(0, 0.05)', linestyle='dotted', linewidth=2,
            color='b')
    ax.plot(line_space, res['scrfp-0.1.txt'], label='SmoothMix + SCRFP-2(0, 0.10)', linestyle='dotted', linewidth=2,
            color='y')

    ax.legend()
    ax.set_xlim([0, 1.6])
    ax.set_ylim([0.0, 0.7])
    ax.set_xlabel('Radius')
    ax.set_ylabel('Accuracy')
    # plt.show()
    plt.savefig('figs/imagenet_0.50.png', bbox_inches='tight')
    print(1)

    ######################################## Noise 1.00 ############################
    base_path = r"E:\Experiments\SCRFP\imagenet\benchmark\noise_1.00"
    names = ['smooth.txt', 'scrfp-0.01.txt', 'scrfp-0.05.txt', 'scrfp-0.1.txt']
    line_space = np.linspace(0, 3.2, 120)

    res = {name: ApproximateAccuracy(os.path.join(base_path, name)).at_radii(line_space) for name in names}
    plt = update_params(plt, {'legend.fontsize': 16, 'axes.labelsize': 22,
                              'xtick.labelsize': 22,
                              'ytick.labelsize': 22, })
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(line_space, res['smooth.txt'], label='Smoothed Classifier', linestyle='solid', linewidth=2, color='r')
    ax.plot(line_space, res['scrfp-0.01.txt'], label='Smoothed Classifier + SCRFP2(0, 0.01)', linestyle='solid',
            linewidth=2, color='g')
    ax.plot(line_space, res['scrfp-0.05.txt'], label='Smoothed Classifier + SCRFP2(0, 0.05)', linestyle='solid',
            linewidth=2, color='b')
    ax.plot(line_space, res['scrfp-0.1.txt'], label='Smoothed Classifier + SCRFP2(0, 0.10)', linestyle='solid',
            linewidth=2, color='y')

    base_path = r"E:\Experiments\SCRFP\imagenet\SmoothAdv\noise_1.00"
    names = ['smooth.txt', 'scrfp-0.01.txt', 'scrfp-0.05.txt', 'scrfp-0.1.txt']

    res = {name: ApproximateAccuracy(os.path.join(base_path, name)).at_radii(line_space) for name in names}
    ax.plot(line_space, res['smooth.txt'], label='SmoothAdv', linestyle='dashed', linewidth=2, color='r')
    ax.plot(line_space, res['scrfp-0.01.txt'], label='SmoothAdv + SCRFP2(0, 0.01)', linestyle='dashed', linewidth=2,
            color='g')
    ax.plot(line_space, res['scrfp-0.05.txt'], label='SmoothAdv + SCRFP2(0, 0.05)', linestyle='dashed', linewidth=2,
            color='b')
    ax.plot(line_space, res['scrfp-0.1.txt'], label='SmoothAdv + SCRFP2(0, 0.10)', linestyle='dashed', linewidth=2,
            color='y')

    base_path = r"E:\Experiments\SCRFP\imagenet\mix\noise_1.00"
    names = ['smooth.txt', 'scrfp-0.01.txt', 'scrfp-0.05.txt', 'scrfp-0.1.txt']

    res = {name: ApproximateAccuracy(os.path.join(base_path, name)).at_radii(line_space) for name in names}
    ax.plot(line_space, res['smooth.txt'], label='SmoothMix', linestyle='dotted', linewidth=2, color='r')
    ax.plot(line_space, res['scrfp-0.01.txt'], label='SmoothMix + SCRFP2(0, 0.01)', linestyle='dotted', linewidth=2,
            color='g')
    ax.plot(line_space, res['scrfp-0.05.txt'], label='SmoothMix + SCRFP-2(0, 0.05)', linestyle='dotted', linewidth=2,
            color='b')
    ax.plot(line_space, res['scrfp-0.1.txt'], label='SmoothMix + SCRFP-2(0, 0.10)', linestyle='dotted', linewidth=2,
            color='y')

    ax.legend()
    ax.set_xlim([0, 3.2])
    # ax.set_ylim([0.2, 0.65])
    ax.set_xlabel('Radius')
    ax.set_ylabel('Accuracy')
    # plt.show()
    plt.savefig('figs/imagenet_1.00.png', bbox_inches='tight')
    print(1)
