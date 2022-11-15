import os

import matplotlib.pyplot as plt
import wandb

from core.engine.dataloader import set_dataset
from core.engine.trainer import set_pl_model
from core.pattern import FloatHook
from core.utils import *
from settings.test_setting import TestParser

if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'activation2']
    args = TestParser(argsv).get_args()
    # os.environ["WANDB_DIR"] = args.model_dir
    api = wandb.Api(timeout=60)
    runs = api.runs(args.project)
    names = ['relu', 'relu6', 'sigmoid', 'lrelu', 'gelu']
    all_float = {}
    for cur_run in runs:
        if cur_run.name not in names:
            continue
        args = TestParser(argsv).get_args()