import os

import matplotlib.pyplot as plt
import wandb

from core.engine.dataloader import set_dataset
from core.engine.trainer import set_pl_model
from core.pattern import FloatHook
from core.utils import *
from settings.test_setting import TestParser

if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'adv_compare']
    args = TestParser(argsv).get_args()
    # os.environ["WANDB_DIR"] = args.model_dir
    api = wandb.Api(timeout=60)
    runs = api.runs(args.project)

    names = ['std', 'noise_0.12', 'noise_0.25', 'noise_0.50', 'fgsm_04', 'fgsm_08']
    all_float = {}
    for cur_run in runs:
        if cur_run.name not in names:
            continue
        args = TestParser(argsv).get_args()
        for k, v in cur_run.config.items():
            if not hasattr(args, k):
                setattr(args, k, v)

        model = set_pl_model(cur_run.config['train_mode'])(args)

        run_path = os.path.join(*cur_run.path)
        name = cur_run.name
        root = os.path.join(args.model_dir, cur_run.id)
        file_path = wandb.restore('ckpt-best.ckpt', run_path=run_path, root=root).name
        model.load_state_dict(torch.load(file_path)['state_dict'])

        model = model.cuda()
        hook = FloatHook(model, Gamma=set_gamma(args.activation))
        hook.set_up()
        model.eval()
        _, test = set_dataset(args)
        sigma = 4 / 255
        float_ratio = []

        for i in range(100):
            for j in range(2):
                x = test[i][0].repeat(500, 1, 1, 1).cuda()
                noise = torch.sign(torch.randn_like(x).to(x.device)) * sigma
                p = model(x + noise)
            float_ratio.append(hook.retrieve())
        all_float[name] = float_ratio
    print(1)

    # names = ['std', 'noise_0.12', 'noise_0.25', 'noise_0.50']
    # names = ['std', 'noise_0.12', 'noise_0.25', 'noise_0.50']
    names = {'std': 'Standard', 'noise_0.12': 'Noise: 0.12', 'noise_0.25': 'Noise: 0.25',
             'noise_0.50': 'Noise: 0.50', 'fgsm_04': 'FGSM: 4/255', 'fgsm_08': 'FGSM: 8/255'}
    fig, ax = plt.subplots(figsize=(18, 8))
    width = 0.15
    plt.style.use('ggplot')
    for i, (name_id, name) in enumerate(names.items()):
        data = np.array(all_float[name_id])
        mean = np.array(data).mean(axis=0)
        xx = np.arange(len(mean)) + (i - 2) * width

        ax.bar(xx, mean, width=width, label=name)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    fig.legend(loc='upper right', bbox_to_anchor=(0.90, 0.88), prop={'size': 12})
    plt.show()
    plt.savefig(bbox_inches='tight')
    # ax.bar(np.arange(0, len(all_mean[0]), 1), all_mean)
