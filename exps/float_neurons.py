from core.pattern import FloatHook, set_output_hook
from core.utils import *
from engine.dataloader import set_dataset
from engine.trainer import PLModel
from settings.train_settings import TrainParser

if __name__ == '__main__':
    sigma = 2 / 255
    argsv = ['--exp_id', 'express2', '--net', 'vgg16', '--model_type', 'mini', '--dataset', 'cifar10']
    args = TrainParser(argsv).get_args()
    path = r'/home/orange/Main/Experiment/TPAMI/vgg16_express2/wandb/run-20221030_202505-22rqzpgl/files/ckpt-best-v1.ckpt'

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
