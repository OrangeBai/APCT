from core.pattern import FloatHook, set_output_hook
from core.utils import *
from core.engine import set_dataset
from core.engine import PLModel
from settings.train_settings import TrainParser
import wandb

if __name__ == '__main__':
    WANDB_DIR = r'/home/orange/Main/Experiment/TPAMI/vgg16_express2/'
    sigma = 4 / 255
    api = wandb.Api()
    runs = api.runs("orangebai/express2")
    for run in runs:
        if run.name == 'batchsize: 512':
            break
    else:
        raise NameError

    argsv = ['--exp_id', 'express2', '--net', 'vgg16', '--model_type', 'mini', '--dataset', 'cifar10']
    args = TrainParser(argsv).get_args()
    # path = os.path.join(WANDB_DIR, 'wandb', run.id, 'files', 'ckpt-best-v1.ckpt')
    path = r'/home/orange/Main/Experiment/TPAMI/vgg16_express2/wandb/run-20221030_202518-1b6rs29h/files/ckpt-best-v1.ckpt'
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
