import math

from settings.base_parser import *
import os
import yaml

from config import *


class TrainParser(BaseParser):
    def __init__(self, argv=None):
        super(TrainParser, self).__init__(argv)
        self._init_parser()

        self.check()

    def _init_parser(self):
        # train mode
        self.parser.add_argument('--train_mode', default='std', type=str)
        self.train_mode()

        # step-wise or epoch-wise
        self.parser.add_argument('--num_epoch', default=None, type=int)
        self.parser.add_argument('--num_step', default=-1, type=int)
        self.parser.add_argument('--batch_size', default=128, type=int)

        # model settings
        self.parser.add_argument('--data_bn', type=int, default=1)
        self.parser.add_argument('--batch_norm', default=1, type=int)
        self.parser.add_argument('--activation', default='LeakyReLU', type=str)
        # scheduler and optimizer
        self.parser.add_argument('--lr_scheduler', default='milestones',
                                 choices=['static', 'milestones', 'exp', 'linear', 'cyclic'])
        self.parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
        self.parser.add_argument('--lr', default=0.1, type=float)

        # training settings
        self.parser.add_argument('--num_workers', default=4, type=int)
        self.parser.add_argument('--npbar', default=True, action='store_false')

        self.lr_scheduler()
        self.optimizer()
        self.dataset()
        return self.parser

    def get_args(self):
        args = self.parser.parse_known_args(self.args)[0]
        return args

    def lr_scheduler(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.lr_scheduler == 'milestones':
            self.parser.add_argument('--gamma', default=0.1, type=float)
            self.parser.add_argument('--milestones', default=[0.5, 0.75], nargs='+', type=float)  # for milestone
        elif args.lr_scheduler in ['exp', 'linear']:
            self.parser.add_argument('--lr_e', default=0.0001 * args.lr, type=float)  # for linear
        elif args.lr_scheduler == 'cyclic':
            self.parser.add_argument('--base_lr', default=0.0001 * args.lr)
            self.parser.add_argument('--up_ratio', default=1 / 3)
            self.parser.add_argument('--down_ratio', default=2 / 3)
        else:
            raise NameError('Scheduler {} not found'.format(args.lr_scheduler))
        return

    def optimizer(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.optimizer == 'SGD':
            # SGD parameters
            self.parser.set_defaults(lr=0.1)
            self.parser.add_argument('--weight_decay', default=5e-4, type=float)
            self.parser.add_argument('--momentum', default=0.9, type=float)
        elif args.optimizer == 'Adam':
            self.parser.set_defaults(lr=0.01)
            self.parser.add_argument('--beta_1', default=0.9, type=float)
            self.parser.add_argument('--beta_2', default=0.99, type=float)
            self.parser.add_argument('--weight_decay', default=5e-4, type=float)
        else:
            pass
        return

    def set_up_attack(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.attack.lower() == 'fgsm':
            self.parser.add_argument('--ord', default='inf')
            self.parser.add_argument('--eps', default=4/255, type=float)
        elif args.attack.lower() == 'pgd':
            self.parser.add_argument('--ord', default='inf')
            self.parser.add_argument('--alpha', default=2/255, type=float)
            self.parser.add_argument('--eps', default=4/255, type=float)
        elif args.attack.lower() == 'noise':
            self.parser.add_argument('--sigma', default=0.12, type=float)
        return

    def dataset(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.dataset.lower() == 'mnist':
            self.parser.set_defaults(model_type='dnn')
            self.parser.add_argument('--num_cls', default=10, type=int)
            self.parser.add_argument('--input_size', default=784, type=int)
            self.parser.add_argument('--width', default=1000, type=int)
            self.parser.add_argument('--depth', default=9, type=int)
        elif args.dataset.lower() == 'cifar10':
            self.parser.add_argument('--num_cls', default=10, type=int)
            self.parser.set_defaults(model_type='mini')
        elif args.dataset.lower() == 'cifar100':
            self.parser.add_argument('--num_cls', default=100, type=int)
            self.parser.set_defaults(model_type='mini')
        elif args.dataset.lower() == 'imagenet':
            self.parser.set_defaults(model_type='net')
            self.parser.add_argument('--num_cls', default=1000, type=int)
        return

    def train_mode(self):
        args, _ = self.parser.parse_known_args(self.args)
        # Prune training
        if args.train_mode == 'std':
            pass
        elif args.train_mode == 'adv':
            self.parser.add_argument('--attack', default='vanilla', type=str)
            self.set_up_attack()
        elif args.train_mode == 'exp':
            self.parser.add_argument('--split', default=1.0, type=float)
        elif args.train_mode == 'pru':
            self.parser.add_argument('--prune_eta', default=1, type=float)
            self.parser.add_argument('--prune_every', default=20, type=float)
            self.parser.add_argument('--method', default='Hard', type=str,
                                     choices=['L1Unstructured', 'RandomStructured', 'LnStructured',
                                              'RandomUnstructured', 'Hard'])
            self.parser.add_argument('--total_amount', default=0.5, type=float)
            self.set_prune()

        return

    def check(self):
        args, _ = self.parser.parse_known_args(self.args)
        if (args.num_epoch is None and args.num_step == -1) or (args.num_epoch is not None and args.num_step != -1):
            raise ValueError('Specify either number of epoch or number of step')
        elif args.num_epoch is not None and args.num_step == -1:
            # if train for certain epochs
            self.parser.add_argument('--val_epoch', default=1, type=int)
            self.parser.add_argument('--val_step', default=None, type=int)
        elif args.num_epoch is None and args.num_step != -1:
            # if train for certain steps
            self.parser.add_argument('--val_epoch', default=None, type=int)
            self.parser.add_argument('--val_step', default=100, type=int)

    def set_prune(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.method == 'Hard':
            self.parser.set_defaults(prune_eta=0)
            self.parser.add_argument('--conv_pru_bound', default=0.1, type=float)
            self.parser.add_argument('--fc_pru_bound', default=0.1, type=float)
        else:
            prune_times = args.num_epoch // args.prune_every
            amount = 1 - math.pow(1 - args.total_amount, 1/prune_times)
            self.parser.add_argument('--amount', default=amount, type=float)

