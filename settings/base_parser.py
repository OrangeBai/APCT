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
        # model type

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
        path = os.path.join(MODEL_PATH, args.dataset, args.net, args.project)
        os.makedirs(path, exist_ok=True)
        self.parser.add_argument('--model_dir', default=path, type=str, help='model directory')
        return

    def get_args(self):
        args = self.parser.parse_known_args(self.args)[0]
        return args
