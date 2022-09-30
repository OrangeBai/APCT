import torch.cuda
import os
from settings.test_setting import TestParser
from models.base_model import build_model
from dataloader.base import *
from core.utils import accuracy, MetricLogger
from core.pattern import *
from exps.smoothed import *
from exps.text_acc import test_acc


if __name__ == '__main__':
    argsv = ['--test_name', 'smoothed_certify']
    torch.cuda.device_count()
    args = TestParser(argsv).get_args()

    model = build_model(args).cuda()
    ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'))
    model.load_weights(ckpt['model_state_dict'])
    # _, test_loader = set_loader(args)
    model.eval()
    smooth_test(model, args)
    test_acc(model, args)




