from settings.test_setting import TestParser
from core.tester import *
from core.tester import BaseTester, DualTester
from core.dual_net import DualNet
import wandb
import torch


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
    print(1)

    # WANDB_DIR = args.model_dir
    # api = wandb.Api()
    # api.runs(args.project, filters={"name": "test"})
    model = torch.load(r"E:\Experiments\model3.pth")
    model = model.eval()
    _, val_set = set_dataloader(args)
    attack = set_attack(model, args)
    api = wandb.Api()
    api.runs(args.project, filters={"display_name": "test"})
    dual_net = DualNet(model, args)
    for x, y in val_set:
        adv = attack.forward(x, y)
        pre_1 = model(x)
        dual_net.predict(torch.concat([x, adv]), 1, 0)

        print(1)
    # for name, module in model.named_modules():
    #     if hasattr(module, 'weight'):
    #         try:
    #             prune.remove(module, 'weight')
    #         except:
    #             pass
    # print(1)
    # model = build_model(args).cuda()
    # ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'))
    # model.load_weights(ckpt['model_state_dict'])
    # # _, test_loader = set_loader(args)
    # model.eval()
    # smooth_test(model, args)
    # test_acc(model, args)




