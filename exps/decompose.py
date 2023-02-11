import torch
from settings.test_setting import TestParser
from core.tester import BaseTester, DualTester, restore_runs
from numpy.linalg import norm
from torch.nn.functional import one_hot, cosine_similarity
from core.dataloader import set_dataset
from core.utils import accuracy
import matplotlib.pyplot as plot

if __name__ == '__main__':
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'adv_compare']
    load_args = TestParser(load_argsv).get_args()
    run_dirs = restore_runs(load_args)

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'test',
             '--test_mode', 'adv', '--attack', 'fgsm', '--eps', '0.0313', '--batch_size', '128']
    args = TestParser(argsv).get_args()
    dual_tester = DualTester(run_dirs, args)
    dual_tester.test(restart=False)

    _, test_set = set_dataset(args)
    ground = one_hot(torch.tensor(test_set.targets)).float()
    keys = list(dual_tester.results.keys())
    keys.sort()
    for k in keys:
        val = dual_tester.results[k]
        arrays = val['array']
        adv_std_diff = arrays[3] - arrays[1]
        fix_diff = arrays[2] - arrays[0]
        float_diff = adv_std_diff - fix_diff
        print("fix_diff similarity: {0}".format(cosine_similarity(torch.tensor(fix_diff).float(), ground)))
        print("fix_diff similarity: {0}".format(cosine_similarity(torch.tensor(float_diff).float(), ground)))
        # std_fix_to_std = norm(arrays[3] - arrays[2], axis=-1) / norm(arrays[:, 0], axis=-1)
        # print("{0}:\nadv_std_diff:{1}\nfix_diff:{2}\nfloat_diff{3}".format(key, adv_std_diff.mean(), fix_diff.mean(), float_diff.mean()))
        print(1)


    _, test_set = set_dataset(args)
    #
    base_tester = BaseTester(run_dirs, args)
    base_tester.test(restart=False)
    print(1)
