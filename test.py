from settings.test_setting import TestParser
# from exps.smoothed import *
# from exps.text_acc import test_acc
import wandb

if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg', '--project', 'test']
    args = TestParser(argsv).get_args()
    WANDB_DIR = args.model_dir
    api = wandb.Api()
    api.runs(args.project, filters={"name": "test"})
    # model = build_model(args).cuda()
    # ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'))
    # model.load_weights(ckpt['model_state_dict'])
    # # _, test_loader = set_loader(args)
    # model.eval()
    # smooth_test(model, args)
    # test_acc(model, args)




