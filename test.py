import torch.cuda

from settings.test_setting import TestParser
from models.base_model import build_model
from dataloader.base import *
from core.utils import accuracy, MetricLogger


if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--exp_id', '0']
    torch.cuda.device_count()
    args = TestParser(argsv).get_args()

    model = build_model(args)
    _, test_loader = set_loader(args)
    model.eval()
    metrics = MetricLogger()
    for images, labels in test_loader:
        images, labels = images.to(0, non_blocking=True), labels.to(0, non_blocking=True)
        # with torch.no_grad():
        # print(images.shape)
        pred = model(images)

        top1, top5 = accuracy(pred, labels)
        metrics.update(top1=(top1, len(images)))



