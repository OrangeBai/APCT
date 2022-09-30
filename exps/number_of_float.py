from config import *
from core.pattern import *
from exps.smoothed import *
from models.base_model import build_model
from settings.test_setting import TestParser


def get_float_ratio(argsv):
    batch_size = 1
    sigma = 0.25
    args = TestParser(argsv).get_args()
    model = build_model(args).cuda()

    _, test_dataset = set_data_set(args)

    ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'))
    model.load_weights(ckpt['model_state_dict'])
    float_hook = ModelHook(model, set_pattern_hook, [0])
    mean, std = [torch.tensor(d).view(len(d), 1, 1) for d in set_mean_sed(args)]
    model.eval()
    all_flt = []
    for i in range(0, len(test_dataset), 10):
        x, _ = test_dataset[i]
        x = x.cuda()
        for j in range(1):
            batch = x.repeat((batch_size + 1, 1, 1, 1))
            n = torch.randn_like(batch).to(x.device) * sigma
            n[0] = 0
            batch_x = batch + n
            batch_x = (batch_x - mean.to(batch_x.device)) / std.to(batch_x.device)
            label = model(batch_x)
            cur_flt = float_hook.retrieve_res(unpack)

            cur_flt_ratio = []
            for block in cur_flt:
                fixed = (block[0][0] == block[0][1]).astype(float)
                cur_flt_ratio.append(fixed.mean())
            all_flt.append(cur_flt_ratio)
    return all_flt


if __name__ == '__main__':
    a = [
        ['--dataset', 'cifar10', '--exp_id', 'std', '--model_type', 'mini', '--net', 'vgg16', '--data_size', '352'],
        ['--dataset', 'cifar10', '--exp_id', 'noise_005', '--model_type', 'mini', '--net', 'vgg16', '--data_size',
         '352'],
        ['--dataset', 'cifar10', '--exp_id', 'noise_010', '--model_type', 'mini', '--net', 'vgg16', '--data_size',
         '352'],
        ['--dataset', 'cifar10', '--exp_id', 'noise_025', '--model_type', 'mini', '--net', 'vgg16', '--data_size',
         '352']
    ]
    xx = []
    for b in a:
        xx.append(get_float_ratio(b))
    print(1)
    exp_dir = os.path.join(MODEL_PATH, 'exp')
    os.makedirs(exp_dir, exist_ok=True)
    n = np.array(xx)
    np.save(os.path.join(exp_dir, 'float_ratio_vgg_sigma_0.25'), n)
