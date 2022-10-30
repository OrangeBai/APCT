from settings.base_parser import *
import os
import shutil

import yaml

from config import *


class TrainParser(BaseParser):
    def __init__(self, argv=None):
        super(TrainParser, self).__init__(argv)
        self.file_setting()
        self._init_parser()
        self.rewrite()

        self.attack()
        self.save()

        # path = os.path.join(self.get_args().model_dir, 'args.yaml')
        # self.modify_parser(path)

    def _init_parser(self):
        self.parser.add_argument('--resume', default=0, type=int)
        # step-wise or epoch-wise
        self.parser.add_argument('--start_epoch', default=0, type=int)
        self.parser.add_argument('--num_epoch', default=None, type=int)
        self.parser.add_argument('--num_step', default=None, type=int)
        self.parser.add_argument('--batch_size', default=128, type=int)

        # DDP
        self.parser.add_argument('--world_size', default=1, type=int)
        # model settings
        self.parser.add_argument('--data_bn', type=int, default=1)
        self.parser.add_argument('--batch_norm', default=1, type=int)
        self.parser.add_argument('--activation', default='LeakyReLU', type=str)
        # trainer settings
        self.parser.add_argument('--train_mode', default='std', type=str)
        self.parser.add_argument('--val_mode', default='std', type=str)
        # scheduler and optimizer
        self.parser.add_argument('--lr_scheduler', default='milestones',
                                 choices=['static', 'milestones', 'exp', 'linear', 'cyclic'])
        self.parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
        self.parser.add_argument('--lr', default=0.1, type=float)
        self.parser.add_argument('--warmup', default=2, type=float)

        # training settings
        self.parser.add_argument('--num_workers', default=1, type=int)
        self.parser.add_argument('--print_every', default=100, type=int)
        self.parser.add_argument('--save_name', default='', type=str)
        # for model pruning
        self.parser.add_argument('--config', default=None)

        # for adv training
        self.parser.add_argument('--attack', default='vanilla', type=str)
        # dataset and experiments
        # gpu settings
        self.parser.add_argument('--cuda', default=[0], type=list)
        self.parser.add_argument('--local_rank', type=int, default=0)
        self.parser.add_argument('--node_rank', type=int, default=0)
        # for debugging
        self.parser.add_argument('--mode', default='client')
        self.parser.add_argument('--port', default=52162)

        # for expressive ability
        self.parser.add_argument('--split', default=1.0, type=float)
        self.parser.add_argument('--pack_every', default=20, type=int)
        self.parser.add_argument('--npbar', default=True, action='store_false')
        self.parser.add_argument('--val_every', default=200, type=int)
        self.resume()
        self.lr_scheduler()
        self.optimizer()
        self.model_type()
        self.dataset()
        self.train_mode()
        return self.parser

    def get_args(self):
        args = self.parser.parse_known_args(self.args)[0]
        if args.lr == 0:
            args.lr += 1e-5
        return args

    def resume(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.resume:
            self.parser.add_argument('--resume_name', default='best')
        return

    def lr_scheduler(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.lr_scheduler == 'milestones':
            self.parser.add_argument('--gamma', default=0.2, type=float)
            self.parser.add_argument('--milestones', default=[0.3, 0.6, 0.8], nargs='+', type=float)  # for milestone
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

    def model_type(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.net == 'dnn':
            self.parser.set_defaults(model_type='dnn')
            self.parser.add_argument('--input_size', default=784, type=int)
            self.parser.add_argument('--width', default=1000, type=int)
            self.parser.add_argument('--depth', default=9, type=int)
        return

    def attack(self):
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
            self.parser.add_argument('--num_cls', default=10, type=int)
        elif args.dataset.lower() == 'cifar10':
            self.parser.add_argument('--num_cls', default=10, type=int)
            self.parser.set_defaults(model_type='mini')
        elif args.dataset.lower() == 'cifar100':
            self.parser.add_argument('--num_cls', default=100, type=int)
            self.parser.set_defaults(model_type='mini')
        elif args.dataset.lower() == 'imagenet':
            self.parser.add_argument('--num_cls', default=1000, type=int)
            self.parser.add_argument('--phase_path', default='./cfgs/phase_default.yml', type=str)
            self.parser.set_defaults(model_type='net')
        return

    def model_dir(self):
        """
        set up the name of experiment: dataset_net_exp_id
        @return:
        """
        args, _ = self.parser.parse_known_args(self.args)
        exp_name = '_'.join([str(args.net), str(args.exp_id)])
        path = os.path.join(MODEL_PATH, args.dir, args.dataset, exp_name)
        self.parser.add_argument('--model_dir', default=path, type=str, help='model directory')
        return self.parser

    def rewrite(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.local_rank != 0 or args.node_rank != 0:
            return
        path = args.model_dir
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)

    def modify_parser(self, file_path):
        cur_args, _ = self.parser.parse_known_args(self.args)
        # Load configuration from yaml file
        with open(file_path, 'r') as file:
            args_dict = yaml.load(file, Loader=yaml.FullLoader)

        for key, val in args_dict.items():
            if '--' + key not in self.args:
                self.args += ['--' + key, str(val)]
        return

    def save(self):
        args, _ = self.parser.parse_known_args(self.args)
        json_file = os.path.join(args.model_dir, 'args.yaml')
        args_dict = vars(args)
        with open(json_file, 'w') as f:
            yaml.dump(args_dict, f)
        return

    def load(self):
        args, _ = self.parser.parse_known_args(self.args)
        json_file = os.path.join(args.model_dir, 'args.yaml')
        self.modify_parser(json_file)
        return

    def file_setting(self):
        self.parser.add_argument('--yml_file', default=None, type=str)
        args, _ = self.parser.parse_known_args(self.args)
        if args.yml_file is not None:
            self.modify_parser(args.yml_file)
        return

    def train_mode(self):
        args, _ = self.parser.parse_known_args(self.args)
        # Prune training
        self.parser.add_argument('--eta_dn', default=0, type=float)
        self.parser.add_argument('--dn_rate', default=0.90, type=float)

        # Certifiable training
        self.parser.add_argument('--eta_fixed', default=0, type=float)
        self.parser.add_argument('--eta_float', default=0, type=float)
        self.parser.add_argument('--noise_type', default='noise', type=str)
        self.parser.add_argument('--noise_eps', default=0.06, type=float)

        # Adversarial Training

        # Prune Training
        self.parser.add_argument('--conv_dn_rate', default=0.95, type=float)
        self.parser.add_argument('--linear_dn_rate', default=0.90, type=float)
        self.parser.add_argument('--prune_every', default=30, type=float)

        # other settings
        self.parser.add_argument('--record_lip', default=1, type=float)
