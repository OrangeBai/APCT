import torch.cuda
import os
from settings.test_setting import TestParser
from models.base_model import build_model
from dataloader.base import *
from core.utils import accuracy, MetricLogger


if __name__ == '__main__':
    argsv = []
    torch.cuda.device_count()
    args = TestParser(argsv).get_args()

    model = build_model(args).to(2)
    ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'),map_location=torch.device(2))
    model.load_weights(ckpt['model_state_dict'])
    _, test_loader = set_loader(args)
    model.eval()
    metrics = MetricLogger()
    for images, labels in test_loader:
        images, labels = images.to(2, non_blocking=True), labels.to(2, non_blocking=True)
        # with torch.no_grad():
        # print(images.shape)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred = model(images)

        top1, top5 = accuracy(pred, labels)
        metrics.update(top1=(top1, len(images)))



