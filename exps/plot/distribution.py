import re
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
from core.engine.trainer import set_pl_model
from settings.test_setting import TestParser
import wandb


if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg', '--project', 'express_04']
    args = TestParser(argsv).get_args()
    WANDB_DIR = args.model_dir
    api = wandb.Api()
    runs = api.runs(args.project, filters={"display_name": {"$regex": "split*"}})
    for i in range(15):
        tag = 'trainset/entropy/layer/{}'.format(str(i).zfill(2))
        all_dist = []
        names = []
        for run in runs:
            all_dist.append(run.history(keys=[tag])[tag][1])
            names.append(run.name)
        fig, ax = plt.subplots()
        ax.hist(all_dist)
        ax.legend(names)
        ax.set_title('Trainset Layer {}'.format(i))
        plt.show()

    for i in range(15):
        tag = 'trainset/entropy/layer/{}'.format(str(i).zfill(2))
        all_dist = []
        names = []
        for run in runs:
            all_dist.append(run.history(keys=[tag])[tag][1])
            names.append(run.name)
        fig, ax = plt.subplots()
        ax.hist(all_dist)
        ax.legend(names)
        ax.set_title('Trainset Layer {}'.format(i))
        plt.show()

#
#     logtool = WandbLogger(name=args.name, save_dir=args.model_dir, project=args.project, config=args)
# sns.set()
# sns.set_theme(style="darkgrid")
#
# large = 22
# med = 16
# small = 12
# params = {'axes.titlesize': large,
#           'legend.fontsize': large,
#           'figure.figsize': (16, 9),
#           'axes.labelsize': large,
#           'xtick.labelsize': large,
#           'ytick.labelsize': large,
#
#           'figure.titlesize': large}
# plt.rcParams.update(params)
# # Project is specified by <entity/project-name>
# numeric_const_pattern = r"""
#      [-+]? # optional sign
#      (?:
#          (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
#          |
#          (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
#      )
#      # followed by optional exponent part if desired
#      (?: [Ee] [+-]? \d+ ) ?
#      """
# rx = re.compile(numeric_const_pattern, re.VERBOSE)
# WANDB_DIR = "/home/orange/Main/Experiment/ICLR/cifar10/cifar10/vgg16_express2/"
# api = wandb.Api()
# runs = api.runs("orangebai/express2")
#
# # bench_id = ''
# # for run in runs:
# #     if run.name == 'split: 100%':
# #         bench_id = run.id
# #         break
#
# batch_ids = {}
# for run in runs:
#     if 'split' in run.name and 'batchsize' in run.name and run.state == 'finished' and run.id != 'wftmzrgd':
#         split, batch_size = rx.findall(run.name)
#         batch_ids[int(float(split) * 100)] = run.id
#         print(run.name)
#
# lines = []
# color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# for i in [1, 6,  11]:
#     fig, ax = plt.subplots()
#     entropy_name = 'val/entropy_layer_{:02}'.format(i)
#     print(entropy_name)
#     keys = list(batch_ids.keys())
#     keys.sort()
#     for k, c in zip(keys, color):
#         run = api.run("orangebai/express2/{}".format(batch_ids[k]))
#         data = run.history(keys=["step", entropy_name, entropy_name])
#         a = list(data[entropy_name])[-1]
#         b = list(data[entropy_name])[-2]
#
#
#         plt.hist([a,b],
#                  histtype='bar',
#                  stacked=False,
#                  fill=True,
#                  # label=labels,
#                  alpha=0.8,  # opacity of the bars
#                  # color=colors,
#                  edgecolor="k")
#
# print(1)