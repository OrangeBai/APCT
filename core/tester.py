import os
import torch
import wandb
import numpy as np
from core.dataloader import set_dataloader
from core.dual_net import DualNet
from core.utils import MetricLogger, accuracy
from core.attack import set_attack


def restore_runs(args, filters):
    api = wandb.Api(timeout=120)
    runs = api.runs(args.project, filters=filters)
    run_paths = {}
    for run in runs:
        run_path = '/'.join(run.path)
        root = os.path.join(args.model_dir, run.id)
        for file in run.files():
            wandb.restore(file.name, run_path=run_path, root=root)
        run_paths[run.name] = os.path.join(root, root)
    return run_paths


class BaseTester:
    def __init__(self, run_dirs, args):
        self.args = args
        self.run_dirs = run_dirs
        self.results = {}

    @staticmethod
    def load_model(run_dir):
        model_path = os.path.join(run_dir, 'model.pth')
        model = torch.load(model_path)
        return model.cuda().eval()

    def test(self):
        for run_name, run_dir in self.run_dirs.items():
            model = self.load_model(run_dir)
            self.results[run_name] = self.test_model(model)
        return

    def test_model(self, model):
        metrics = MetricLogger()
        _, val_loader = set_dataloader(self.args)
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                pred = model(images)
            top1, top5 = accuracy(pred, labels)
            metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))
        return {meter: metrics.retrieve_meters(meter).avg for meter in metrics.meters}


class DualTester(BaseTester):
    def __init__(self, model, args):
        super(DualTester, self).__init__(model, args)
        self.model = model
        self.dual_net = DualNet(model, args)

    def test_model(self, model):
        metrics = MetricLogger()
        dual_net = DualNet(model, self.args).eval()
        _, val_loader = set_dataloader(self.args)
        attack = set_attack(model, self.args).eval()
        all_pred = []
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            adv_images = attack.forward(images, labels).detach()

            pred = torch.stack([self._decompose_path(dual_net, image, adv) for image, adv in zip(images, adv_images)])
            adv_pred = model(adv_images)

            top1, top5 = accuracy(pred[:, 0], labels)
            top1_adv, top5_adv = accuracy(adv_pred, labels)
            top1_fix_std, top5_fix_std = accuracy(pred[:, 1], labels)
            top1_fix_adv, top5_fix_adv = accuracy(pred[:, 2], labels)

            metrics.update(top1=(top1, len(images)), top5=(top5, len(images)),
                           top1_adv=(top1_adv, len(images)), top5_adv=(top5_adv, len(images)),
                           top1_fix_std=(top1_fix_std, len(images)), top5_fix_std=(top5_fix_std, len(images)),
                           top1_fix_adv=(top1_fix_adv, len(images)), top5_fix_adv=(top5_fix_adv, len(images))
                           )
            all_pred.append(pred.detach().cpu().numpy())

        result = {meter: metrics.retrieve_meters(meter).avg for meter in metrics.meters}
        result['array'] = np.concatenate(all_pred)
        return result

    @staticmethod
    def _decompose_path(dual_net, image, adv):
        pred_std, pred_fix = dual_net.predict(torch.stack([image, adv]), 0, -1)
        masked_pred = dual_net.masked_predict(torch.unsqueeze(image, dim=0), dual_net.fixed_neurons, 0, -1)[0]
        return torch.stack([pred_std, masked_pred, pred_fix])

