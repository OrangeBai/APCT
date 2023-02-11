import os

import matplotlib.pyplot as plt
import wandb

from core.dataloader import set_dataset
from core.trainer import set_pl_model
from core.utils import *
from settings.test_setting import TestParser


if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg16', '--project', 'adv_compare']
    args = TestParser(argsv).get_args()
    # os.environ["WANDB_DIR"] = args.model_dir
    api = wandb.Api(timeout=60)
    runs = api.runs(args.project)

    names = ['std', 'noise_0.12', 'noise_0.25', 'noise_0.50', 'fgsm_04', 'fgsm_08']
    all_lip = {}
    all_dist = {}
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
        # hook = FloatHook(model, Gamma=set_gamma(args.activation))
        # hook.set_up()
        model.eval()
        _, test = set_dataset(args)
        sigma = 8 / 255
        lip_point = []
        dist = []
        for i in range(500):
            x = test[i][0].repeat(500, 1, 1, 1).cuda()
            y = torch.tensor(test[i][1]).repeat(500).cuda()
            noise = torch.sign(torch.randn_like(x).to(x.device)) * sigma
            noised_x = x + noise
            noised_x.requires_grad = True
            p = model(noised_x)
            cost = torch.nn.CrossEntropyLoss()(p, y)
            grad = torch.autograd.grad(cost, noised_x, retain_graph=False, create_graph=False)[0]

            p2 = model(noised_x + grad)
            lip_norm = grad.view(len(grad), -1).norm(p=2, dim=-1)
            lip = (p2 - p).norm(p=2, dim=-1) / lip_norm
            adv = torch.sign(grad) * 4 / 255
            p3 = model(noised_x + adv)
            lip_point.append(lip.detach().cpu().numpy())
            dist.append((p3 - p2).norm(p=2, dim=1).detach().cpu().numpy())
        all_lip[name] = lip_point
        all_dist[name] = dist
    print(1)

#   Plot Lip Mean and Dist Mean
    names = {'std': 'Standard', 'noise_0.12': 'Noise: 0.12', 'noise_0.25': 'Noise: 0.25',
             'noise_0.50': 'Noise: 0.50', 'fgsm_04': 'FGSM: 4/255', 'fgsm_08': 'FGSM: 8/255'}
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    plt.style.use('ggplot')
    for name_idx, name in names.items():
        lip = np.array(all_lip[name_idx])
        lip_var = (lip / lip[:, 0][:, np.newaxis]).var(axis=1)
        lip_mean = lip.mean(axis=1)

        dist = np.array(all_dist[name_idx])
        dist_mean = dist.mean(axis=1)
        dist_var = (dist / dist[:, 0][:, np.newaxis]).var(axis=1)
        ax.scatter(np.log(lip_mean), np.log(dist_mean), label=name, s=12)
    fig.legend(loc='lower right', bbox_to_anchor=(0.90, 0.1), prop={'size': 18})
    ax.set_xlabel('log(Mean of Lipschitz Constant)', fontsize=18)
    ax.set_ylabel('log(Mean of Prediction Distance)', fontsize=18)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    # plt.show()
    plt.savefig('figs/lip_mean_dist_mean', bbox_inches='tight')

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    plt.style.use('ggplot')
    for name_idx, name in names.items():
        lip = np.array(all_lip[name_idx])
        lip_mean = lip.mean(axis=1)
        lip_ratio = (lip / lip[:, 0][:, np.newaxis])
        lip_var = lip_ratio.var(axis=1)

        dist = np.array(all_dist[name_idx])
        dist_mean = dist.mean(axis=1)
        dist_ratio = dist / dist[:, 0][:, np.newaxis]
        dist_var = dist.var(axis=1)
        # dist_var = (dist_ratio * dist_ratio).sum(axis=1) / 499
        ax.scatter(np.log(lip_mean), lip_var, label=name, s=18)
    fig.legend(loc='upper right', bbox_to_anchor=(0.90, 0.88), prop={'size': 18})
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    ax.set_xlabel('log(Mean of Lipschitz Constant)', fontsize=18)
    ax.set_ylabel('Variance of Lipschitz Constant', fontsize=18)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 1)
    plt.savefig('figs/lip_mean_lip_var', bbox_inches='tight')

    # Lip mean and Lip var
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    plt.style.use('ggplot')
    for name_idx, name in names.items():
        lip = np.array(all_lip[name_idx])
        lip_mean = lip.mean(axis=1)
        lip_ratio = (lip / lip[:, 0][:, np.newaxis])
        lip_var = lip_ratio.var(axis=1)

        dist = np.array(all_dist[name_idx])
        dist_mean = dist.mean(axis=1)
        dist_ratio = dist / dist[:, 0][:, np.newaxis]
        dist_var = dist.var(axis=1)
        # dist_var = (dist_ratio * dist_ratio).sum(axis=1) / 499
        ax.scatter(np.log(lip_mean), dist_var, label=name, s=18)
    fig.legend(loc='upper right', bbox_to_anchor=(0.90, 0.88), prop={'size': 18})
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    ax.set_xlabel('log(Mean of Lipschitz Constant)', fontsize=18)
    ax.set_ylabel('Variance of Prediction Distance', fontsize=18)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 1)
    plt.savefig('figs/lip_mean_dist_var', bbox_inches='tight')

