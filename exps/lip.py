import numpy as np
import torch.nn

from config import *
from core.pattern import *
from exps.smoothed import *
from models.base_model import build_model
from settings.test_setting import TestParser
from exps.text_acc import *


def local_lip(argsv, sigma):
    batch_size = 64
    args = TestParser(argsv).get_args()
    model = build_model(args).cuda()

    _, test_dataset = set_data_set(args)

    ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'))
    model.load_weights(ckpt['model_state_dict'])
    # float_hook = ModelHook(model, set_pattern_hook, [0])
    mean, std = [torch.tensor(d).view(len(d), 1, 1) for d in set_mean_sed(args)]
    model.eval()
    all_flt = []
    # test_acc(model, args)
    m, v = [], []
    for i in range(0, len(test_dataset), 100):
        x, y, = test_dataset[i]
        x, y = x.cuda(), torch.tensor(y).cuda()
        batch = x.repeat((batch_size, 1, 1, 1))
        label = y.repeat((batch_size, 1))
        signal = torch.sign(torch.randn_like(batch).to(x.device))
        n = signal / signal.view(batch_size, -1).norm(p=2, dim=1).view(batch_size, 1, 1, 1) * sigma
        n[0] = 0
        batch_x = batch + n
        batch_x = (batch_x - mean.to(batch_x.device)) / std.to(batch_x.device)
        batch_x.requires_grad = True
        pred = model(batch_x)
        loss = torch.nn.CrossEntropyLoss()(pred, label.squeeze())
        grad = torch.autograd.grad(loss, batch_x, retain_graph=False, create_graph=False)[0]
        grad_view = grad.view(batch_size, -1)
        pred_2 = model(batch_x + grad)

        p = (pred_2 - pred).norm(p=2, dim=1) / grad_view.norm(p=2, dim=1)
        m.append(p.mean().cpu().detach().numpy())
        v.append(p.var().cpu().detach().numpy())
        print(1)
    return m, v

    #     x, _ = test_dataset[i]
    #     x = x.cuda()
    #     x =
    #
    #     torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]


if __name__ == '__main__':
    a = [
        ['--dataset', 'cifar10', '--exp_id', 'std', '--model_type', 'mini', '--net', 'vgg16', '--data_size', '352'],
        ['--dataset', 'cifar10', '--exp_id', 'noise_005', '--model_type', 'mini', '--net', 'vgg16', '--data_size',
         '352'],
        ['--dataset', 'cifar10', '--exp_id', 'noise_010', '--model_type', 'mini', '--net', 'vgg16', '--data_size',
         '352'],
        ['--dataset', 'cifar10', '--exp_id', 'noise_025', '--model_type', 'mini', '--net', 'vgg16', '--data_size',
         '352']
    ]
    xx = []

    for b in a:
        x = []
        for s in range(1, 17):
            x.append(local_lip(b, s))
        xx.append(x)
    print(1)
    exp_dir = os.path.join(MODEL_PATH, 'exp')
    os.makedirs(exp_dir, exist_ok=True)
    n = np.array(xx)
    np.save(os.path.join(exp_dir, 'lip'), n)
