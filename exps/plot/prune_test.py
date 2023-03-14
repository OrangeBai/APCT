import pandas as pd
from settings.test_setting import TestParser
from argparse import Namespace
from core.tester import PruneTester, restore_runs
from exps.plot.plt_base import update_params, update_ax_font
import matplotlib.pyplot as plt


if __name__ == '__main__':
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'prune_test']
    load_args = TestParser(load_argsv).get_args()
    run_dirs = restore_runs(load_args)

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'prune_test',
             '--test_mode', 'prune', '--method', 'LnStructured', '--prune_eta', '2']
    args = TestParser(argsv).get_args()

    tester = PruneTester(run_dirs, args)
    tester.test(restart=True)
    print(1)
