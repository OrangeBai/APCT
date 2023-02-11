import os
import torch
import wandb
import numpy as np
from core.dataloader import set_dataloader
from core.dual_net import DualNet
from core.utils import MetricLogger, accuracy
from core.attack import set_attack
from functools import wraps


# equals to save_and_load(name)(fun)(self, run_dir)
def save_and_load(name):
    def inner(fun):
        def wrapper(*args, **kwargs):
            self, run_dir, restart = args
            result_path = self.result_path(run_dir, name)
            if os.path.exists(result_path) and not restart:
                return torch.load(result_path)
            else:
                result = fun(self, run_dir)
                self.save_test_result(result, result_path)
                return result
        return wrapper
    return inner


def restore_runs(args, filters=None):
    api = wandb.Api(timeout=120)
    runs = api.runs(args.project, filters=filters)
    run_paths = {}
    for run in runs:
        run_path = '/'.join(run.path)
        root = os.path.join(args.model_dir, run.id)
        for file in run.files():
            if not os.path.exists(os.path.join(root, file.name)):
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

    def test(self, restart=False):
        for run_name, run_dir in self.run_dirs.items():
            os.makedirs(os.path.join(run_dir, 'test'), exist_ok=True)
            self.results[run_name] = self.test_model(run_dir, restart)
        return

    @staticmethod
    def save_test_result(result, result_path):
        torch.save(result, result_path)

    @staticmethod
    def result_path(run_dir, name):
        return os.path.join(run_dir, 'test', name + '.pth')

    @staticmethod
    def load_test_result(result_path):
        return torch.load(result_path)

    @save_and_load('acc')
    def test_model(self, run_dir, restart=False):
        model = self.load_model(run_dir)
        metrics = MetricLogger()
        _, val_loader = set_dataloader(self.args)
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                pred = model(images)
            top1, top5 = accuracy(pred, labels)
            metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))
        result = {meter: metrics.retrieve_meters(meter).avg for meter in metrics.meters}
        return result


class DualTester(BaseTester):
    def __init__(self, model, args):
        super(DualTester, self).__init__(model, args)
        self.model = model
        self.dual_net = DualNet(model, args)

    @save_and_load('decompose')
    def test_model(self, run_dir, restart=False):
        model = self.load_model(run_dir)
        metrics = MetricLogger()
        dual_net = DualNet(model, self.args).eval()
        _, val_loader = set_dataloader(self.args)
        attack = set_attack(model, self.args).eval()
        all_pred = []
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            adv_images = attack.forward(images, labels).detach()

            stacked_pred = torch.stack([self._decompose_path(dual_net, image, adv) for image, adv in zip(images, adv_images)])
            adv_pred = model(adv_images)
            pred = torch.concat([stacked_pred.transpose(1, 0), adv_pred.unsqueeze(dim=0)])

            top1, top5 = accuracy(pred[1], labels)
            top1_adv, top5_adv = accuracy(pred[3], labels)
            top1_fix_std, top5_fix_std = accuracy(pred[0], labels)
            top1_fix_adv, top5_fix_adv = accuracy(pred[2], labels)

            metrics.update(top1=(top1, len(images)), top5=(top5, len(images)),
                           top1_adv=(top1_adv, len(images)), top5_adv=(top5_adv, len(images)),
                           top1_fix_std=(top1_fix_std, len(images)), top5_fix_std=(top5_fix_std, len(images)),
                           top1_fix_adv=(top1_fix_adv, len(images)), top5_fix_adv=(top5_fix_adv, len(images))
                           )
            all_pred.append(pred.detach().cpu().numpy())

        result = {meter: metrics.retrieve_meters(meter).avg for meter in metrics.meters}
        result['array'] = np.concatenate(all_pred, axis=1)
        return result

    @staticmethod
    def _decompose_path(dual_net, image, adv):
        pred_std, pred_fix = dual_net.predict(torch.stack([image, adv]), 0, -1)
        masked_pred = dual_net.masked_predict(torch.unsqueeze(image, dim=0), dual_net.fixed_neurons, 0, -1)[0]
        return torch.stack([masked_pred, pred_std, pred_fix])

