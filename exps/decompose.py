from settings.test_setting import TestParser
from core.tester import *
from core.tester import BaseTester, DualTester


if __name__ == '__main__':
    load_argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'test']
    load_args = TestParser(load_argsv).get_args()
    run_dirs = restore_runs(load_args, filters={"display_name": "test"})

    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'test',
             '--test_mode', 'adv', '--attack', 'fgsm', '--eps', '0.0313', '--batch_size', '128']
    args = TestParser(argsv).get_args()
    dual_tester = DualTester(run_dirs, args)
    dual_tester.test()

    base_tester = BaseTester(run_dirs, args)
    base_tester.test()
