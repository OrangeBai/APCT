from dataloader import get_val_loader
from core.utils import accuracy, MetricLogger
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
