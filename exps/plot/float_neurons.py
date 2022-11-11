from core.pattern import FloatHook, set_output_hook
from core.utils import *
from core.engine import set_dataset
from core.engine import PLModel
from settings.test_setting import TestParser
import wandb

if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg', '--project', 'adv_compare']
    args = TestParser(argsv).get_args()
    WANDB_DIR = args.model_dir
    api = wandb.Api(timeout=60)

    model = PLModel.load_from_checkpoint(path, args=args)
    hook = FloatHook(model.model, set_output_hook, Gamma=set_gamma(args.activation))
    model.eval()
    _, test = set_dataset(args)
    for i in range(1000):
        for j in range(2):
            x = test[i][0].repeat(500, 1, 1, 1).cuda()
            noise = torch.sign(torch.randn_like(x).to(x.device)) * sigma
            p = model(x + noise)
        a = hook.retrieve()
    print(1)
