from settings.test_setting import TestParser
from core.trainer import set_pl_model
from torch.nn.utils.prune import global_unstructured, identity
import wandb
import torch
if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'test', '--num_cls', '10']
    args = TestParser(argsv).get_args()
    WANDB_DIR = args.model_dir
    api = wandb.Api()
    api.runs(args.project, filters={"name": "test"})
    file = torch.load(r"F:\PHD\Computer science\_Paper\Ap_Cert\submisson\arxiv\ckpt-best.ckpt")
    model = set_pl_model('std')(args)

    global_unstructured(
        model.model,
        pruning_method=identity,
    )
    print(1)
    # model = build_model(args).cuda()
    # ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'))
    # model.load_weights(ckpt['model_state_dict'])
    # # _, test_loader = set_loader(args)
    # model.eval()
    # smooth_test(model, args)
    # test_acc(model, args)




