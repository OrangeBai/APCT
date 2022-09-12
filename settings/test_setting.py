from settings.train_settings import *


def set_up_testing(argv=None):
    arg_parser = ArgParser(False, argv)
    parser = arg_parser.parser
    args = arg_parser.args

    parser.add_argument('--test_name', default='test_acc', type=str)
    test_name = parser.parse_known_args(args)[0].test_name
    if test_name == 'test_acc':
        parser = test_acc(parser)
    elif test_name.lower() == 'ap_lip':
        parser = ap_lip(parser)
    elif test_name.lower() == 'td':
        parser = td(parser)
    elif test_name.lower() == 'smooth':
        parser = smoothed_certify(parser)
    else:
        raise NameError('test name {0} not found'.format(test_name))

    exp_dir = os.path.join(parser.parse_args().model_dir, 'exp')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    parser.add_argument('--exp_dir', default=exp_dir, type=str)
    return parser.parse_args(args)


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
    parser.add_argument("--sigma", type=float, help="noise hyperparameter", default=0.1)
    parser.add_argument("--batch", type=int, default=1000, help="batch size")
    parser.add_argument("--N0", type=int, default=100)
    parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
    parser.add_argument("--smooth_alpha", type=float, default=0.001, help="failure probability")
    return parser


def test_acc(parser):
    parser.set_defaults(ord='inf', alpha=2 / 255, eps=8 / 255)
    return parser
