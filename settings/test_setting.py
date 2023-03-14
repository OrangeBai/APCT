from email.policy import default
from settings.train_settings import *


class TestParser(BaseParser):
    def __init__(self, argv=None):
        super(TestParser, self).__init__(argv)
        self.parser.add_argument('--test_mode', default='acc', type=str)
        self._set_up_test()

    def _set_up_test(self):
        args, _ = self.parser.parse_known_args(self.args)
        if args.test_mode == 'acc':
            pass
        elif args.test_mode == 'adv':
            self.parser.add_argument('--attack', default='fgsm')
            self.set_up_attack()
        elif args.test_mode == 'smoothed_certify':
            self.parser = smoothed_certify(self.parser)
        elif args.test_mode == 'prune':
            args, _ = self.parser.parse_known_args(self.args)
            self.parser.add_argument('--method', default='Hard', type=str)
            self.parser.add_argument('--prune_eta', default=-1, type=int)

            self.parser.add_argument('--conv_bound', default=0.1, type=float)
            self.parser.add_argument('--fc_bound', default=0.1, type=float)

            self.parser.add_argument('--conv_amount', default=0.2, type=float)
            self.parser.add_argument('--fc_amount', default=0.2, type=float)


#
# def set_up_testing(argv=None):
#     arg_parser = TrainParser(False, argv)
#     parser = arg_parser.parser
#     args = arg_parser.args
#
#     parser.add_argument('--test_name', default='test_acc', type=str)
#     test_name = parser.parse_known_args(args)[0].test_name
#     if test_name == 'test_acc':
#         parser = test_acc(parser)
#     elif test_name.lower() == 'ap_lip':
#         parser = ap_lip(parser)
#     elif test_name.lower() == 'td':
#         parser = td(parser)
#     elif test_name.lower() == 'smooth':
#         parser = smoothed_certify(parser)
#     else:
#         raise NameError('test name {0} not found'.format(test_name))
#
#     exp_dir = os.path.join(parser.parse_args().model_dir, 'exp')
#     if not os.path.exists(exp_dir):
#         os.makedirs(exp_dir)
#     parser.add_argument('--exp_dir', default=exp_dir, type=str)
#     return parser.parse_args(args)


def ap_lip(parser):
    parser.add_argument('--epsilon', nargs='+', default=[2 / 255, 4 / 255, 8 / 255, 16 / 255], type=float)
    parser.add_argument('--sample_size', default=256, type=int)
    parser.add_argument('--num_test', default=100)
    return parser


def td(parser):
    parser.add_argument('--line_breaks', default=2048, type=int)
    parser.add_argument('--num_test', default=100, type=int)
    parser.add_argument('--pre_batch', default=512)
    return parser


def non_ret(parser):
    parser.add_argument('--num_test', default=100, type=int)
    parser.add_argument('--line_breaks', default=2048, type=int)
    parser.add_argument('--pre_batch', default=512)
    return parser


def smoothed_certify(parser):
    parser.add_argument('--smooth_model', default='smooth', type=str)
    parser.add_argument("--sigma", type=float, help="noise hyperparameter", default=0.125)
    parser.add_argument("--N0", type=int, default=100)
    parser.add_argument('--skip', type=int, default=10)
    parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument('--method', default='SMRAP', type=str)
    parser.add_argument('--batch', default=256, type=int)
    parser.add_argument('--eta_fixed', default=0.00, type=float)
    parser.add_argument('--eta_float', default=0.00, type=float)
    return parser


def test_acc(parser):
    parser.set_defaults(ord='inf', alpha=2 / 255, eps=8 / 255)
    return parser
