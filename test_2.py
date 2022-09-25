import torch.cuda
import os
from settings.test_setting import TestParser
from models.base_model import build_model
from dataloader.base import *
from core.utils import accuracy, MetricLogger
from core.pattern import *
from exps.smoothed import *


if __name__ == '__main__':
    argsv = ['--dataset', 'imagenet', '--exp_id', 'noise_000', '--model_type', 'net', '--test_name', 'smoothed_certify',
             '--net', 'resnet50']
    torch.cuda.device_count()
    args = TestParser(argsv).get_args()

    model = build_model(args).cuda()
    # torch.load(os.path.join(args.model_dir, ''))

    ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'))
    model.load_weights(ckpt['model_state_dict'])
    # _, test_loader = set_loader(args)
    model.eval()
    smooth_test(model, args)



    # metrics = MetricLogger()
    # for images, labels in test_loader:
    #     images, labels = images.to(2, non_blocking=True), labels.to(2, non_blocking=True)
    #     # with torch.no_grad():
    #     # print(images.shape)
    #     with torch.cuda.amp.autocast(dtype=torch.float16):
    #         pred = model(images)
    #
    #     top1, top5 = accuracy(pred, labels)
    #     metrics.update(top1=(top1, len(images)))



