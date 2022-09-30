from dataloader.base import *
from dataloader.imagenet import get_val_loader
from attack import set_attack
from core.utils import to_device, accuracy, MetricLogger
import torch

def test_acc(model, args):
    if args.dataset.lower() == 'imagenet':
        test_loader = get_val_loader(args)
    else:
        _, test_loader = set_loader(args)
    metrics = MetricLogger()
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            pred = model(images)
        top1, top5 = accuracy(pred, labels)
        metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))
    print(metrics)
    return metrics
