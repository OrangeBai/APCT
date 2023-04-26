import itertools
import os
import pandas as pd
import torch
import wandb
from core.scrfp import ApproximateAccuracy
from core.trainer import BaseTrainer
from settings.test_setting import TestParser
from core.utils import MetricLogger, accuracy
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
from core.dataloader import set_dataloader, set_dataset
from core.trainer import BaseTrainer
if __name__ == '__main__':
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'prune_l1']
    load_args = TestParser(load_argsv).get_args()

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'test',
             '--test_mode', 'acc', '--batch_size', '128']
    args = TestParser(argsv).get_args()
    # runs = restore_runs(load_args)
    # tt = BaseTrainer(args).load_best(r"E:\Experiments\cifar10\vgg16\prune_l1\2sjfaiyv")
    model = torch.load(r"E:\Experiments\cifar10\vgg16\adv_compare\3k5q9z58\model.pth").cuda()

    model.eval()
    metrics = MetricLogger()
    _, val_loader = set_dataloader(args)
    for images, labels in val_loader:
        images, labels = images.cuda(), labels.cuda()
        pred = model(images)
        top1, top5 = accuracy(pred, labels)
        metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))
    print(1)