import datetime
import os

import torch.nn as nn

from core.smooth_analyze import *
from core.smooth_core import *
from core.SMRAP import SMRAP
from dataloader.base import *


def smooth_test(model, args):
    file_path = os.path.join(args.exp_dir, '_'.join([args.method, str(args.N0), str(args.N), str(args.sigma_2)]))
    smooth_pred(model, args)

    certify_res = ApproximateAccuracy(file_path).at_radii(np.linspace(0, 1, 256))
    output_path = os.path.join(args.exp_dir, args.method + '_cert.npy')
    print(certify_res.mean())
    np.save(output_path, certify_res)
    return


def smooth_pred(model, args):
    if args.method == 'SMRAP':
        smoothed_classifier = SMRAP(model, args)
    else:
        smoothed_classifier = Smooth(model, args)

    # prepare output file
    file_path = os.path.join(args.exp_dir, '_'.join([args.method, str(args.N0), str(args.N), str(args.sigma_2)]))
    f = open(file_path, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    _, dataset = set_data_set(args)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == -1:
            break

        (x, label) = dataset[i]

        before_time = time.time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.smooth_alpha, args.batch_size)
        after_time = time.time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
