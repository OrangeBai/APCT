import argparse
import os
import sys

import yaml

from config import *


class BaseParser:
    def __init__(self, argv=None):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', default='cifar10', type=str)
        self.parser.add_argument('--exp_id', default=0, type=str)
        self.parser.add_argument('--dir', default='', type=str)
        # model type
        self.parser.add_argument('--model_type', default='net', choices=['dnn', 'mini', 'net'])
        self.parser.add_argument('--net', default='vgg16', type=str)

        self.unknown_args = []
        if argv is None:
            self.args = sys.argv[1:]
        else:
            self.args = sys.argv[1:] + argv
        self.model_dir()

        # path = os.path.join(self.get_args().model_dir, 'args.yaml')
        # self.modify_parser(path)

    def model_dir(self):
        """
        set up the name of experiment: dataset_net_exp_id
        @return:
        """
        args, _ = self.parser.parse_known_args(self.args)
        exp_name = '_'.join([str(args.net), str(args.exp_id)])
        path = os.path.join(MODEL_PATH, args.dir, args.dataset, exp_name)
        self.parser.add_argument('--model_dir', default=path, type=str, help='model directory')
        return

    def load(self):
        args, _ = self.parser.parse_known_args(self.args)
        json_file = os.path.join(args.model_dir, 'args.yaml')
        self.modify_parser(json_file)
        return

    def save(self):
        args, _ = self.parser.parse_known_args(self.args)
        json_file = os.path.join(args.model_dir, 'args.yaml')
        args_dict = vars(args)
        with open(json_file, 'w') as f:
            yaml.dump(args_dict, f)
        return

    def modify_parser(self, file_path):
        cur_args, _ = self.parser.parse_known_args(self.args)
        # Load configuration from yaml file
        with open(file_path, 'r') as file:
            args_dict = yaml.load(file, Loader=yaml.FullLoader)

        for key, val in args_dict.items():
            if not hasattr(cur_args, key):
                self.parser.add_argument('--'+ key, default=val)

            self.parser.set_defaults(key=val)
        return

    def get_args(self):
        args = self.parser.parse_known_args(self.args)[0]
        return args
