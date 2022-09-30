from collections import defaultdict, deque, Iterable, OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler


class LLR(_LRScheduler):
    def __init__(self, optimizer, lr_st, lr_ed, steps, last_epoch=-1, verbose=False):
        self.lr_st = lr_st
        self.lr_ed = lr_ed
        self.steps = steps
        self.diff = lr_st - lr_ed
        super(LLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [self.lr_st * (self.lr_st - (self.last_epoch / self.steps) * self.diff) / self.lr_st
                for group in self.optimizer.param_groups]


def warmup_scheduler(args, optimizer):
    def lambda_rule(step):
        return step / (args.warmup_steps + 1e-8)

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
    return lr_scheduler


def init_scheduler(args, optimizer):
    """
    Initialize learning rate scheduler.
    Milestone:
            args.milestone: milestones to decrease the learning rate
                        [milestone_1, milestone_2, ..., milestone_n]
            args.gamma: scale factor
            the learning rate is scaled by gamma when iteration reaches each milestone
    Linear:
            args.lr_e: desired learning rate at the end of training
            the learning rate decreases linearly from lr to lr_e
    Exp:
            args.lr_e: desired learning rate at the end of training
            the learning rate decreases exponentially from lr to lr_e
    Cyclic:
            args.up_ratio: ratio of training steps in the increasing half of a cycle
            args.down_ratio: ratio of training steps in the decreasing half of a cycle
            args.lr_e: Initial learning rate which is the lower boundary in the cycle for each parameter group.
    Static:
            the learning rate remains unchanged during the training
    """
    if args.lr == 0:
        args.lr += 1e-6
    if args.lr_scheduler == 'milestones':
        milestones = [milestone * args.total_step for milestone in args.milestones]
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    elif args.lr_scheduler == 'linear':
        # diff = args.lr - args.lr_e
        # LinearLR(optimizer, start_factor=args.lr, end_factor=args.lr_e, total_iters=args.num_)
        # def lambda_rule(step):
        #     return (args.lr - (step / args.total_step) * diff) / args.lr

        lr_scheduler = LLR(optimizer, lr_st=args.lr, lr_ed=args.lr_e, steps=args.total_step)

    elif args.lr_scheduler == 'exp':
        gamma = math.pow(args.lr_e / args.lr, 1 / args.total_step)
        lr_scheduler = ExponentialLR(optimizer, gamma)
    elif args.lr_scheduler == 'cyclic':
        up = int(args.total_step * args.up_ratio)
        down = int(args.total_step * args.down_ratio)
        lr_scheduler = CyclicLR(optimizer, base_lr=args.lr_e, max_lr=args.lr,
                                step_size_up=up, step_size_down=down, mode='triangular2', cycle_momentum=False)
    elif args.lr_scheduler == 'static':
        def lambda_rule(t):
            return 1.0

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
    # TODO ImageNet scheduler
    else:
        raise NameError('Scheduler {0} not found'.format(args.lr_scheduler))
    return lr_scheduler


def init_optimizer(args, model):
    """
    Initialize optimizer:
        SGD: Implements stochastic gradient descent (optionally with momentum).
             args.momentum: momentum factor (default: 0.9)
             args.weight_decay: weight decay (L2 penalty) (default: 5e-4)
        Adam: Implements Adam algorithm.
            args.beta_1, beta_2:
                coefficients used for computing running averages of gradient and its square, default (0.9, 0.99)
            args.eps: term added to the denominator to improve numerical stability (default: 1e-8)
            args.weight_decay: weight decay (L2 penalty) (default: 5e-4)
    """
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr + 1e-8, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr + 1e-8, betas=(args.beta_1, args.beta_2),
                                     weight_decay=args.weight_decay)
    else:
        raise NameError('Optimizer {0} not found'.format(args.lr_scheduler))
    return optimizer


def init_loss(args):
    # TODO other loss functions
    return torch.nn.CrossEntropyLoss()


def accuracy(output, target, top_k=(1, 5)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    @param output: predicted result
    @param target: ground truth
    @param top_k: Object to be computed
    @return: accuracy (in percentage, not decimal)
    """
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def is_dist_avail_and_initialized():
    # TODO review distributed coding
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        """
        @param window_size: Window size for computation of  median and average value
        @param fmt: output string format
        """
        if fmt is None:
            fmt = "{avg:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0  # sum of all instances
        self.count = 0  # number of updates
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([self.total, self.count], dtype=torch.float32, device=device)
        dist.reduce(total, dst=0, op=dist.ReduceOp.SUM, async_op=False)
        self.total, self.count = total.tolist()

    @property
    def median(self):
        # return the median value of in the queue
        d = torch.tensor(np.array(list(self.deque)))
        return d.median().item()

    @property
    def avg(self):
        # average value of the queue
        try:
            d = torch.tensor(np.array(list(self.deque)), dtype=torch.float32)
            return d.mean().item()
        except ValueError:
            return -1

    @property
    def global_avg(self):
        # global average value of the record variable
        return self.total / (self.count + 1e-2)

    @property
    def value(self):
        # latest value of the record variable
        try:
            return self.deque[-1]
        except IndexError:
            return -1

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value)

    def to_dict(self):
        # covert all perperties to a dictionary, for recording purpose
        return {
            'median': self.median,
            'total': self.total,
            'avg': self.avg,
            'global_avg': self.global_avg,
            'count': self.count
        }


class MetricLogger:
    """
    Metric logger: Record the meters (top 1, top 5, loss and time) during the training.
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.iter_time = SmoothedValue(fmt='{global_avg:.4f}')

    def reset(self):
        # reset metric logger
        for name in self.meters.keys():
            self.meters[name].reset()

    def add_meter(self, name, meter):
        # add a meter
        self.meters[name] = meter

    def update(self, **kwargs):
        # update the metric values
        for k, (v, n) in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n)
        self.all_reduce()

    def retrieve_meters(self, k):
        if k in self.meters.keys():
            return self.meters[k]
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, k))

    # def __getattr__(self, attr):
    #     if attr in self.meters:
    #         return self.meters[attr]
    #     if attr in self.__dict__:
    #         return self.__dict__[attr]
    #     raise AttributeError("'{}' object has no attribute '{}'".format(
    #         type(self).__name__, attr))

    def __str__(self):

        loss_str = []

        for name, meter in self.meters.items():
            if len(meter.deque) > 0:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)

    def all_reduce(self):
        #  TODO check out how to use multi-process
        if dist.is_initialized():
            dist.barrier()
            for meter in self.meters.values():
                meter.all_reduce()


def to_device(device_id=None, *args):
    """
    Send the *args variables to designated device
    @param device_id: a valid device id
    @param args: a number of numbers/models
    @return:
    """
    if device_id is None:
        return args
    else:
        return [arg.cuda(device_id) for arg in args]


def check_activation(layer):
    acts = [nn.LeakyReLU, nn.ReLU, nn.ELU, nn.Sigmoid, nn.GELU, nn.Tanh, nn.PReLU]
    for ll in acts:
        if isinstance(layer, ll):
            return True
    return False


def to_numpy(tensor):
    if type(tensor) == np.ndarray:
        return tensor
    else:
        return tensor.cpu().detach().numpy()


def set_gamma(activation):
    if activation.lower() in ['relu', 'prelu', 'gelu', 'leakyrelu']:
        return [0]
    elif activation.lower() == 'sigmoid':
        return [-0.5, 0.5]


def set_lb_ub(activation):
    if activation.lower() == 'relu':
        return (0, 0), (1, 1)
    elif activation.lower() == 'prelu':
        return (0.1, 0.1), (1, 1)
    elif activation.lower() == 'gelu':
        return (0.1, 0.1), (1, 1)
    elif activation.lower() == 'leakyrelu':
        return (0.1, 0.1), (1, 1)
    elif activation.lower() == 'sigmoid':
        return (0, 0.2), (0.2, 0.5), (0, 0.2)


def check_phase(phase_file, epoch):
    for p, v in phase_file.items():
        if epoch in range(v['start_epoch'], v['end_epoch']):
            return p, v
    raise ValueError('Phase file not matching training')


class ImageNetScheduler():
    def __init__(self, optimizer, phases):
        self.optimizer = optimizer
        self.current_lr = None
        self.phases = [self.format_phase(p) for p in phases]
        self.tot_epochs = max([max(p['ep']) for p in self.phases])

    def format_phase(self, phase):
        phase['ep'] = listify(phase['ep'])
        phase['lr'] = listify(phase['lr'])
        if len(phase['lr']) == 2:
            assert (len(phase['ep']) == 2), 'Linear learning rates must contain end epoch'
        return phase

    def linear_phase_lr(self, phase, epoch, batch_curr, batch_tot):
        lr_start, lr_end = phase['lr']
        ep_start, ep_end = phase['ep']
        if 'epoch_step' in phase: batch_curr = 0  # Optionally change learning rate through epoch step
        ep_relative = epoch - ep_start
        ep_tot = ep_end - ep_start
        return self.calc_linear_lr(lr_start, lr_end, ep_relative, batch_curr, ep_tot, batch_tot)

    def calc_linear_lr(self, lr_start, lr_end, epoch_curr, batch_curr, epoch_tot, batch_tot):
        step_tot = epoch_tot * batch_tot
        step_curr = epoch_curr * batch_tot + batch_curr
        step_size = (lr_end - lr_start) / step_tot
        return lr_start + step_curr * step_size

    def get_current_phase(self, epoch):
        for phase in reversed(self.phases):
            if (epoch >= phase['ep'][0]): return phase
        raise Exception('Epoch out of range')

    def get_lr(self, epoch, batch_curr, batch_tot):
        phase = self.get_current_phase(epoch)
        if len(phase['lr']) == 1: return phase['lr'][0]  # constant learning rate
        return self.linear_phase_lr(phase, epoch, batch_curr, batch_tot)

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_lr(epoch, batch_num, batch_tot)
        if self.current_lr == lr: return
        if ((batch_num == 1) or (batch_num == batch_tot)):
            log.event(f'Changing LR from {self.current_lr} to {lr}')

        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        tb.log("sizes/lr", lr)
        tb.log("sizes/momentum", args.momentum)


def listify(p=None, q=None):
    if p is None:
        p = []
    elif not isinstance(p, Iterable):
        p = [p]
    n = q if type(q) == int else 1 if q is None else len(q)
    if len(p) == 1: p = p * n
    return p


def load_weight(model, state_dict):
    new_dict = OrderedDict()
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), state_dict.items()):
        if v1.shape == v2.shape:
            new_dict[k1] = v2
        else:
            raise KeyError
    model.load_state_dict(new_dict)
    return model


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


class InputCenterLayer(torch.nn.Module):
    """Centers the channels of a batch of images by subtracting the dataset mean.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(InputCenterLayer, self).__init__()
        self.means = torch.tensor(means).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return input - means