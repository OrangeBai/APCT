import itertools
import os
import pandas as pd
import torch
import wandb
from core.scrfp import ApproximateAccuracy
from settings.test_setting import TestParser
from argparse import Namespace
from core.tester import SmoothedTester, restore_runs
from exps import update_params, update_ax_font
from numpy.linalg import norm
from torch.nn.functional import one_hot, cosine_similarity
from core.dataloader import set_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy

if __name__ == '__main__':
    base_path = r'E:\Experiments\SCRFP\imagenet\mix\noise_1.00'
    scrfp_10_path = os.path.join(base_path, 'scrfp-0.01.txt')
    scrfp_05_path = os.path.join(base_path, 'scrfp-0.05.txt')
    scrfp_25_path = os.path.join(base_path, 'scrfp-0.1.txt')
    smooth_path = os.path.join(base_path, 'smooth.txt')

    scrfp_10 = ApproximateAccuracy(scrfp_10_path).at_radii(np.linspace(0, 2.95, 13))
    scrfp_05 = ApproximateAccuracy(scrfp_05_path).at_radii(np.linspace(0, 2.95, 13))
    scrfp_25 = ApproximateAccuracy(scrfp_25_path).at_radii(np.linspace(0, 2.95, 13))
    smooth = ApproximateAccuracy(smooth_path).at_radii(np.linspace(0, 2.95, 13))

    print(1)

