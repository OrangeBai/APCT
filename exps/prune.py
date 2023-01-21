import os

import matplotlib.pyplot as plt
import wandb
from settings.test_setting import TestParser

if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'prune_ln']
    args = TestParser(argsv).get_args()
    # os.environ["WANDB_DIR"] = args.model_dir
    api = wandb.Api(timeout=60)
    runs = api.runs(args.project)

    print(1)
    # file_path = wandb.restore('ckpt-best.ckpt', run_path=run_path, root=root).name