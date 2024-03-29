import argparse
import os
import sys

import yaml

from config import *


class BaseParser:
    def __init__(self, argv=None):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', type=str, required=True)
        self.parser.add_argument('--net', type=str, required=True)
        self.parser.add_argument('--project', type=str, required=True)
        self.parser.add_argument('--name', type=str)
        # model settings
        self.parser.add_argument('--batch_norm', default=1, type=int)
        self.parser.add_argument('--activation', default='ReLU', type=str)

        # data loader settings
        self.parser.add_argument('--batch_size', default=128, type=int)
        self.parser.add_argument('--num_workers', default=4, type=int)

        self.unknown_args = []
        if argv is None:
            self.args = sys.argv[1:]
        else:
            self.args = sys.argv[1:] + argv
        self.model_dir()
        self.dataset()

        # path = os.path.join(self.get_args().model_dir, 'args.yaml')
        # self.modify_parser(path)

    def model_dir(self):
        """
        set up the name of experiment: dataset_net_exp_id
        @return:
        """
        args, _ = self.parser.parse_known_args(self.args)
        path = os.path.join(MODEL_PATH, args.dataset, args.net, args.project)
        os.makedirs(path, exist_ok=True)
        self.parser.add_argument('--model_dir', default=path, type=str, help='model directory')
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

    def set_up_attack(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.attack.lower() == 'fgsm':
            self.parser.add_argument('--ord', default='inf')
            self.parser.add_argument('--eps', default=4 / 255, type=float)
        elif args.attack.lower() == 'pgd':
            self.parser.add_argument('--ord', default='inf')
            self.parser.add_argument('--alpha', default=2 / 255, type=float)
            self.parser.add_argument('--eps', default=4 / 255, type=float)
        elif args.attack.lower() == 'noise':
            self.parser.add_argument('--sigma', default=0.12, type=float)
        return

    def set_prune(self):
        args, _ = self.parser.parse_known_args(self.args)
        milestones = list(range(args.prune_every, args.num_epoch - args.fine_tune + 1, args.prune_every))
        self.parser.add_argument('--prune_milestones', default=milestones, type=int, nargs='+')
        if args.method == 'Hard':
            self.parser.set_defaults(prune_eta=-1)
            self.parser.add_argument('--conv_bound', default=0.1, type=float)
            self.parser.add_argument('--fc_bound', default=0.1, type=float)
        else:
            prune_times = len(milestones)
            self.parser.set_defaults(conv_amount=args.conv_amount / prune_times)
            self.parser.set_defaults(fc_amount=args.fc_amount / prune_times)

    def get_args(self):
        args = self.parser.parse_known_args(self.args)[0]
        return args
