import os
import torch
import wandb
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import wandb
from settings.test_setting import TestParser
from core.tester import *
from core.tester import BaseTester, DualTester


if __name__ == '__main__':
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'express2']

    WANDB_DIR = "/home/orange/Main/Experiment/ICLR/cifar10/cifar10/vgg16_express2/"
    api = wandb.Api(timeout=300)
    run = api.run("orangebai/express2/wftmzrgd")
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    fig, axes = plt.subplots(2, 2, figsize=(24, 10))
    layers = [2, 5, 8, 11]

    for idx, layer_i in enumerate(layers):
        name = "val/entropy_layer_" + str(layer_i).zfill(2)
        i, j = idx // 2, idx % 2
        pd_10 = run.history(keys=["step", name])
        xx = np.linspace(0, 6000, 121)
        yy = np.array(list(pd_10[name])).T
        yy = yy[:1024, :]
        print(xx.shape)
        print(yy.shape)
        axes[i][j].plot(xx, yy.T, color="C0", alpha=0.075)

        axes[i][j].spines['top'].set_visible(False)
        axes[i][j].spines['right'].set_visible(False)
        axes[i][j].spines['bottom'].set_visible(False)
        axes[i][j].spines['left'].set_visible(False)

        axes[i][j].tick_params(axis='x', labelsize=24)
        axes[i][j].tick_params(axis='y', labelsize=24)
        axes[i][j].set_ylabel('float entropy', fontsize=24)
        axes[i][j].set_xlabel('steps', fontsize=24)
        axes[i][j].set_title('Layer: ' + '{}'.format(layer_i).rjust(2), fontsize=24)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        wspace=0.1,
                        hspace=0.5)
    plt.savefig('plot/figs/float_entropy_dynamic.png', bbox_inches='tight')
