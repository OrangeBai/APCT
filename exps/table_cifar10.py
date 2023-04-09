import os
import pandas as pd
import wandb
from core.scrfp import ApproximateAccuracy
from settings.test_setting import TestParser
from argparse import Namespace
from core.tester import restore_runs
from numpy.linalg import norm
import seaborn as sns
import numpy as np
from copy import deepcopy

if __name__ == '__main__':
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16']
    load_args = TestParser(load_argsv).get_args()

    runs = restore_runs(load_args)

    for sigma in ['0.125', '0.25', '0.5']:
        argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--test_mode', 'smoothed_certify',
                 '--smooth_model', 'smooth', '--sigma', sigma]
        args = TestParser(argsv).get_args()
        test_names = [i.format(args.sigma) for i in ['flt_{}_0.01', 'flt_{}_0.02',  'flt_{}_0.05', 'flt_{}_0.10', 'std_{}']]
        run_dirs = {run: run_dir for run, run_dir in runs.items() if run.name in test_names}
        res = {}
        for n, p in run_dirs.items():
            smooth_path = os.path.join(p, 'test', 'smooth.txt')
            scrfp_path = os.path.join(p, 'test', 'scrfp-0.1.txt')
            res['smooth_' + n.name] = list(ApproximateAccuracy(smooth_path).at_radii(np.linspace(0, 2, 9)))
            res['scrfp_' + n.name] = ApproximateAccuracy(scrfp_path).at_radii(np.linspace(0, 2, 9))
        res = {k: res[k] for k in sorted(res)}
        res = pd.DataFrame(res).T
        print((100*res).to_latex(float_format="%.1f"))
