import re
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
from core.engine.trainer import set_pl_model
from settings.test_setting import TestParser
import wandb
import pandas as pd
import numpy as np

if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg', '--project', 'express_05']
    args = TestParser(argsv).get_args()
    WANDB_DIR = args.model_dir
    api = wandb.Api(timeout=60)
    runs = api.runs(args.project, filters={"display_name": {"$regex": "split*"}})

    penguins = sns.load_dataset("penguins")
    train_df, val_df = [], []
    run = api.runs(args.project, filters={"display_name": 'split_0.1'})[0]
    for i in range(0, 15):
        tag = 'trainset/entropy_dist_{}'.format(str(i).zfill(2))
        dist = run.history(keys=[tag])[tag][1]
        df = {'value': dist, 'layer': i, 'dataset': 'train', 'exp': run.name, 'weight': 1/len(dist)}
        train_df.append(pd.DataFrame(df))

        tag = 'entropy/layer/{}'.format(str(i).zfill(2))
        dist = run.history(keys=[tag])[tag][1]
        df = {'value': dist, 'layer': i, 'dataset': 'val', 'exp': run.name, 'weight': 1/len(dist)}
        val_df.append(pd.DataFrame(df))
    train_df, val_df = pd.concat(train_df), pd.concat(val_df)
    sns.histplot(train_df, stat='probability', bins=40, x='layer', y='value', hue='dataset', multiple='layer',
                 pthresh=0.05, cbar=True, weights='weight', legend=False)
    plt.show()
    sns.histplot(val_df, stat='probability', bins=40, x='layer', y='value', hue='dataset', multiple='layer',
                 pthresh=0.05, cbar=True, weights='weight', legend=False)
    plt.show()
    print(1)

    train_df, val_df = [], []
    run = api.runs(args.project, filters={"display_name": 'split_1.0'})[0]
    for i in range(0, 15):
        tag = 'trainset/entropy_dist_{}'.format(str(i).zfill(2))
        dist = run.history(keys=[tag])[tag][1]
        df = {'value': dist, 'layer': i, 'dataset': 'train', 'exp': run.name, 'weight': 1/len(dist)}
        train_df.append(pd.DataFrame(df))

        tag = 'entropy/layer/{}'.format(str(i).zfill(2))
        dist = run.history(keys=[tag])[tag][1]
        df = {'value': dist, 'layer': i, 'dataset': 'val', 'exp': run.name, 'weight': 1/len(dist)}
        val_df.append(pd.DataFrame(df))
    train_df, val_df = pd.concat(train_df), pd.concat(val_df)
    sns.histplot(train_df, stat='probability', bins=40, x='layer', y='value', hue='dataset', multiple='layer',
                 pthresh=0.05, cbar=True, weights='weight', legend=False)
    plt.show()
    sns.histplot(val_df, stat='probability', bins=40, x='layer', y='value', hue='dataset', multiple='layer',
                 pthresh=0.05, cbar=True, weights='weight', legend=False)
    plt.show()
    print(1)



    all_df = []
    run = api.runs(args.project, filters={"display_name": 'split_0.1'})[0]
    for i in range(2, 15):
        tag = 'trainset/entropy_dist_{}'.format(str(i).zfill(2))
        df = {'value': run.history(keys=[tag])[tag][1], 'layer': i, 'dataset': 'train', 'exp': run.name}
        all_df.append(pd.DataFrame(df))

        tag = 'entropy/layer/{}'.format(str(i).zfill(2))
        df = {'value': run.history(keys=[tag])[tag][1], 'layer': i, 'dataset': 'val', 'exp': run.name}
        all_df.append(pd.DataFrame(df))
    all_df = pd.concat(all_df)
    sns.histplot(all_df, stat='probability', bins=35, x='layer', y='value', hue='dataset', multiple='layer',
                 pthresh=0.05, cbar=True)
    plt.show()
    print(1)
        # fig, ax = plt.subplots()
        # ax.hist(all_dist)
        # ax.legend(names)
        # ax.set_title('Trainset Layer {}'.format(i))
        # plt.show()

    for i in range(10,15):
        tag = 'entropy/layer/{}'.format(str(i).zfill(2))
        all_dist = []
        names = []
        dist = run.history(keys=[tag])[tag].iloc[-1]
        all_dist.append(dist)
        names.append(run.name)
        fig, ax = plt.subplots()
        ax.hist(all_dist)
        ax.legend(names)
        ax.set_title('Val set Layer {}'.format(i))
        plt.show()


    run = api.runs(args.project, filters={"display_name": 'split_0.1'})[0]
    all_df = {}
    for i in range(10, 15):
        tag = 'entropy/layer/{}'.format(str(i).zfill(2))
        dist = run.history(keys=[tag])[tag].iloc[-1]
        all_df[i] = dist
    all_df = pd.DataFrame(all_df)
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