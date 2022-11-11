from core.pattern import FloatHook
from core.utils import *
from core.engine.dataloader import set_dataset
from core.engine.trainer import set_pl_model
from pytorch_lightning.loggers import WandbLogger
from settings.test_setting import TestParser
import wandb
import os


if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'adv_compare']
    args = TestParser(argsv).get_args()
    # os.environ["WANDB_DIR"] = args.model_dir
    api = wandb.Api(timeout=60)
    runs = api.runs(args.project)

    cur_run = runs[2]
    for k, v in cur_run.config.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    model = set_pl_model(cur_run.config['train_mode'])(args)
    restore = wandb.restore('ckpt-best.ckpt', run_path=os.path.join(*cur_run.path), root=os.path.join(args.model_dir, cur_run.id)).name
    model.load_state_dict(torch.load(restore)['state_dict'])
    model = model.cuda()
    hook = FloatHook(model, Gamma=set_gamma(args.activation))
    hook.set_up()
    model.eval()
    _, test = set_dataset(args)
    sigma = 4/255
    for i in range(1000):
        for j in range(2):
            x = test[i][0].repeat(500, 1, 1, 1).cuda()
            noise = torch.sign(torch.randn_like(x).to(x.device)) * sigma
            p = model(x + noise)
        a = hook.retrieve()
    print(1)
