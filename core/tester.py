import os
import torch
import wandb
import numpy as np
from numpy.linalg import norm as norm
from core.dataloader import set_dataloader, set_dataset
from core.dual_net import DualNet
from core.utils import MetricLogger, accuracy
from core.attack import set_attack
from core.scrfp import Smooth, SCRFP, ApproximateAccuracy
from shutil import rmtree
from torch.nn.functional import one_hot, cosine_similarity
from core.pattern import FloatHook, set_gamma
import pandas as pd
import time
import datetime
from core.pattern import PruneHook
from core.prune import prune_block, iteratively_prune
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
        run_paths[run] = root
    ids = [run.id for run in runs]
    remove_dirs = [d for d in os.listdir(args.model_dir) if d not in ids]
    for d in remove_dirs:
        rmtree(os.path.join(args.model_dir, d))
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
        for run, run_dir in self.run_dirs.items():
            os.makedirs(os.path.join(run_dir, 'test'), exist_ok=True)
            self.results[run.name] = self.test_model(run_dir, restart)
            self.upload(run, run_dir)
        return self.results

    @staticmethod
    def upload(run, run_dir):
        os.chdir(run_dir)
        test_dir = os.path.join(run_dir, 'test')
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                run.upload_file(path=os.path.join(test_dir, file))
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
    def __init__(self, run_dirs, args):
        super(DualTester, self).__init__(run_dirs, args)

    @save_and_load('noise_dist')
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

            stacked_pred = torch.stack(
                [self._decompose_path(dual_net, image, adv) for image, adv in zip(images, adv_images)])
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

    @staticmethod
    def compute_fix_flt_diff(arrays):
        adv_std_diff = arrays[3] - arrays[1]
        fix_diff = arrays[2] - arrays[0]
        flt_diff = adv_std_diff - fix_diff
        return fix_diff, flt_diff

    def _cal_dist_dist(self, val):
        fix_diff, flt_diff = self.compute_fix_flt_diff(val['array'])
        return pd.DataFrame({'fix': np.linalg.norm(fix_diff, axis=1, ord=2),
                             'flt': np.linalg.norm(flt_diff, axis=1, ord=2)})

    def cal_angle_dist(self, k, val, ground_truth):
        fix_diff, flt_diff = self.compute_fix_flt_diff(val['array'])
        sim_fix = cosine_similarity(torch.tensor(fix_diff).float(), ground_truth).numpy()
        sim_flt = cosine_similarity(torch.tensor(flt_diff).float(), ground_truth).numpy()
        ['fixed', ] * len(fix_diff) + ['float', ] * len(flt_diff)
        return pd.DataFrame({'value': np.concatenate([sim_fix, sim_flt]),
                             'path_type': ['fixed', ] * len(fix_diff) + ['float', ] * len(flt_diff),
                             'name': [k, ] * len(fix_diff) + [k, ] * len(flt_diff)})

    def format_angel_result(self, keys, legend_keys):
        _, test_set = set_dataset(self.args)
        ground = one_hot(torch.tensor(test_set.targets)).float()
        return pd.concat([self.cal_angle_dist(k, self.results[k], ground) for k, legend_k in zip(keys, legend_keys)])

    def format_dist_result(self, keys, legend_keys):
        return {legend_k: self._cal_dist_dist(self.results[k]) for k, legend_k in zip(keys, legend_keys)}

    def float_fixed_vulnerable(self, keys, legend_keys):
        return {legend_k: self._flt_fix_vulnerable(self.results[k]) for k, legend_k in zip(keys, legend_keys)}

    def _flt_fix_vulnerable(self, v):
        _, test_set = set_dataset(self.args)
        ground = one_hot(torch.tensor(test_set.targets)).float().numpy()
        fix, flt = self.compute_fix_flt_diff(v['array'])
        incorrect = (np.argmax(v['array'][3], axis=1) != test_set.targets) * (np.argmax(v['array'][1], axis=1) == test_set.targets)
        fix_per, flt_per = (fix * ground).sum(axis=1), (flt * ground).sum(axis=1)
        # return ((fix_per - flt_per)[incorrect] > 0).sum(axis=0), ((fix_per - flt_per)[incorrect] < 0).sum(axis=0)
        return sum((fix_per - flt_per) > 0)


class FltRatioTester(BaseTester):
    def __init__(self, run_dirs, args):
        super().__init__(run_dirs, args)

    @save_and_load('flt_ratio')
    def test_model(self, run_dir, restart=False):
        model = self.load_model(run_dir)
        model.eval()
        hook = FloatHook(model, Gamma=set_gamma(self.args.activation))
        hook.set_up()
        _, test = set_dataset(self.args)
        sigma = 8 / 255
        float_ratio = []
        t0 = time.time()
        for i in range(500):
            if i % 100 == 0 and i > 0:
                print("Current {0}, avg time: {1}".format(i, (time.time() - t0) / i))
            x = test[i][0].repeat(500, 1, 1, 1).cuda()
            noise = torch.sign(torch.randn_like(x).to(x.device)) * sigma
            model(x + noise)
            float_ratio.append(hook.retrieve())
        return np.array(float_ratio)


class LipNormTester(BaseTester):
    def __init__(self, run_dirs, args):
        super().__init__(run_dirs, args)

    @save_and_load('lip_vul')
    def test_model(self, run_dir, restart=False):
        model = self.load_model(run_dir)
        _, val_loader = set_dataloader(self.args)
        attack = set_attack(model, self.args).eval()
        result = []
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            adv_images = attack.forward(images, labels).detach()
            pred_std = model(images)
            pred_adv = model(adv_images)
            input_diff = adv_images - images
            fixed_diff = (model(images + input_diff / 1e3) - pred_std) * 1e3
            float_diff = pred_adv - fixed_diff
            result.append(np.array([fixed_diff.detach().cpu().numpy(), float_diff.detach().cpu().numpy()]))
            if idx > 100:
                break
        return result


class AveragedFloatTester(BaseTester):
    def __init__(self, run_dirs, args):
        super().__init__(run_dirs, args)

    @save_and_load('noise_dist')
    def test_model(self, run_dir, restart=False):
        model = self.load_model(run_dir)
        dual_net = DualNet(model, self.args).eval()
        _, test = set_dataset(self.args)
        attack = set_attack(model, self.args).eval()
        model = model.cuda()
        model.eval()
        total_pred_diff = []
        for i in range(2):
            x = test[i][0].repeat(1, 1, 1, 1).cuda()
            y = torch.tensor(test[i][1]).repeat(1).cuda()
            y_pred = model(x)
            adv = attack.forward(x, y).detach()
            adv = adv.repeat(256, 1, 1, 1).cuda()

            pred_diff = []
            for _ in range(10):
                noise = torch.randn_like(adv).to(adv.device) * 0.125
                noise[0] = 0
                noised_adv = adv + noise
                pred_diff.append([(y_pred - dual_net.predict(noised_adv, 0, -j * 0.05)).cpu().detach().numpy()[1:] for j in range(5)])
                # norm(y_pred - dual_net.predict(noised_adv, 0, -0.5).detach().cpu().numpy(), ord=2, axis=1).mean()
            cur_repressed = np.concatenate(pred_diff, axis=1)
            # norm(cur_repressed, ord=2, axis=2).mean(1)
            total_pred_diff.append(cur_repressed)
        return np.array(total_pred_diff)

    def compute_norm(self):
        return {k: norm(v, axis=-1, ord=1).mean(axis=(0, 2)) for k, v in self.results.items()}

class LipDistTester(BaseTester):
    def __init__(self, run_dirs, args):
        super().__init__(run_dirs, args)

    @save_and_load('lip_dist')
    def test_model(self, run_dir, restart=False):
        model = self.load_model(run_dir)
        model = model.cuda()
        model.eval()
        _, test = set_dataset(self.args)
        sigma = 8 / 255
        lip_point = []
        dist = []
        for i in range(10000):
            x = test[i][0].repeat(256, 1, 1, 1).cuda()
            y = torch.tensor(test[i][1]).repeat(256).cuda()
            noise = torch.sign(torch.randn_like(x).to(x.device)) * sigma
            noised_x = x + noise
            noised_x.requires_grad = True
            p = model(noised_x)
            cost = torch.nn.CrossEntropyLoss()(p, y)
            grad = torch.autograd.grad(cost, noised_x, retain_graph=False, create_graph=False)[0]

            p2 = model(noised_x + grad)
            lip_norm = grad.view(len(grad), -1).norm(p=2, dim=-1)
            lip = (p2 - p).norm(p=2, dim=-1) / lip_norm
            adv = torch.sign(grad) * sigma

            p3 = model(noised_x + adv)

            lip_point.append(lip.detach().cpu().numpy())
            dist.append((p3 - p2).norm(p=2, dim=1).detach().cpu().numpy())

        return {'all_lip': lip_point, 'all_dist': dist}

    def get_mean_var(self, target):  # get desired variable
        def v(x):
            return np.array(x['all_lip']) if target == 'lip' else np.array(x['all_dist'])

        mean = {k: np.array(v(val)).mean(axis=1) for k, val in self.results.items()}
        var = {k: (v(val) / v(val)[:, 0][:, np.newaxis]).var(axis=1) for k, val in self.results.items()}
        return mean, var


class RobustnessTester(BaseTester):
    def __init__(self, run_dirs, args, namespaces=None):
        super().__init__(run_dirs, args)
        self.namespaces = namespaces

    @save_and_load('robust')
    def test_model(self, run_dir, restart=False):
        model = self.load_model(run_dir)
        metrics = MetricLogger()
        attacks = {name: set_attack(model, namespace) for name, namespace in self.namespaces.items()}

        _, val_loader = set_dataloader(self.args)
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                pred = model(images)
            top1, top5 = accuracy(pred, labels)
            metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))
            for name, attack in attacks.items():
                adv_images = attack.forward(images, labels).detach()
                adv_pred = model(adv_images)
                top1, top5 = accuracy(adv_pred, labels)
                kwargs = {name + '_top1': (top1, len(images)), name + '_top5': (top5, len(images))}
                metrics.update(**kwargs)
        result = {meter: metrics.retrieve_meters(meter).global_avg for meter in metrics.meters}
        return result


class SmoothedTester(BaseTester):
    def __init__(self, run_dirs, args):
        super().__init__(run_dirs, args)

    @save_and_load('smooth')
    def test_model(self, run_dir, restart=False):
        args = self.args
        model = self.load_model(run_dir)
        if args.smooth_model == 'smooth':
            smoothed_classifier = Smooth(model, self.args)
            file_path = os.path.join(run_dir, 'test', 'smooth.txt')
        else:
            smoothed_classifier = SCRFP(model, self.args)
            file_path = os.path.join(run_dir, 'test', 'scrfp.txt')
        # create the smooothed classifier g

        # prepare output file
        f = open(file_path, 'w')
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

        # iterate through the dataset
        _, dataset = set_dataset(self.args)
        for i in range(len(dataset)):

            # only certify every args.skip examples, and stop after args.max examples
            if i % self.args.skip != 0:
                continue

            (x, label) = dataset[i]

            before_time = time.time()
            # certify the prediction of g around x
            x = x.cuda()
            prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
            after_time = time.time()
            correct = int(prediction == label)

            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

        f.close()
        return ApproximateAccuracy(file_path).at_radii(np.linspace(0, 2, 100))


class PruneTester(BaseTester):
    def __init__(self, run_dirs, args):
        super().__init__(run_dirs, args)

    # @save_and_load('prune')
    # def test_model(self, run_dir, restart=False):
    #     model = self.load_model(run_dir)
    #     prune_hook = PruneHook(model, set_gamma(self.args.activation), 0.1)
    #     train_loader, val_loader = set_dataloader(self.args)
    #     for images, labels in train_loader:
    #         images, labels = images.cuda(), labels.cuda()
    #         _ = model(images)
    #         global_entropy = prune_hook.retrieve(reshape=False)
    #
    #         im_scores = {}
    #         for name, block in model.named_modules():
    #             if check_block(model, block):
    #                 im_scores.update(prune_block(block, global_entropy[name], self.args.prune_eta))
    #         iteratively_prune(im_scores, self.args)
    #
    #     metrics = MetricLogger()
    #     for images, labels in val_loader:
    #         images, labels = images.cuda(), labels.cuda()
    #         with torch.no_grad():
    #             pred = model(images)
    #         top1, top5 = accuracy(pred, labels)
    #         metrics.update(top1=(top1, len(images)), top5=(top5, len(images)))
    #     result = {meter: metrics.retrieve_meters(meter).avg for meter in metrics.meters}
    #     return result
