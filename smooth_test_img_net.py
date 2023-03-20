import itertools
import os
import pandas as pd
import torch
import wandb
from core.scrfp import ApproximateAccuracy
from settings.test_setting import TestParser
from core.utils import MetricLogger, accuracy
from core.dataloader import set_dataset, set_dataloader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy
import time
from core.scrfp import Smooth, SCRFP
from models.base_model import build_model
import datetime

if __name__ == '__main__':
    argsv = ['--dataset', 'imagenet', '--net', 'resnet50', '--test_mode', 'smoothed_certify',
             '--smooth_model', 'SCRFP', '--project', 'test']
    args = TestParser(argsv).get_args()
    model = build_model(args)

    model_path = os.path.join(args.path, str(int(100 * args.sigma)))
    model.load_weights(torch.load(os.path.join(model_path, 'model.pth.tar'))['state_dict'])

    model = model.cuda()
    model.eval()
    _, dataset = set_dataset(args)

    if args.smooth_model == 'smooth':
        smoothed_classifier = Smooth(model, args)
        file_path = os.path.join(model_path, 'smooth.txt')
    else:
        smoothed_classifier = SCRFP(model, args)
        file_path = os.path.join(model_path, 'scrfp' + str(args.eta_float) + '.txt')
    # create the smooothed classifier g

    # prepare output file
    f = open(file_path, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    # iterate through the dataset
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % 25 != 0:
            continue

        (x, label) = dataset[i]

        before_time = time.time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch_size)
        after_time = time.time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()

    print(1)


