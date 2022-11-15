import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

from settings.test_setting import TestParser

if __name__ == '__main__':
    argsv = ['--dataset', 'cifar10', '--net', 'vgg', '--project', 'express_split']
    args = TestParser(argsv).get_args()
    WANDB_DIR = args.model_dir
    api = wandb.Api(timeout=60)
    runs = api.runs(args.project, filters={"display_name": {"$regex": "split*"}})

    plt.style.use('ggplot')
    train_df, val_df = [], []
    run = api.runs(args.project, filters={"display_name": 'split_1.00'})[0]
    for i in range(0, 15):
        tag = 'train_set/entropy_dist/layer_{}'.format(str(i).zfill(2))
        dist = run.history(keys=[tag])[tag][1]
        df = {'value': dist, 'layer': i, 'dataset': 'train', 'exp': run.name, 'weight': 1 / len(dist)}
        train_df.append(pd.DataFrame(df))

        tag = 'val_set/entropy_dist/layer_{}'.format(str(i).zfill(2))
        dist = run.history(keys=[tag])[tag][1]
        df = {'value': dist, 'layer': i, 'dataset': 'val', 'exp': run.name, 'weight': 1 / len(dist)}
        val_df.append(pd.DataFrame(df))
    train_df, val_df = pd.concat(train_df), pd.concat(val_df)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax = sns.histplot(train_df, stat='probability', bins=60, x='layer', y='value', hue='dataset', multiple='layer',
                      pthresh=0.03, cbar=True, weights='weight', legend=False, cbar_kws=dict(shrink=.75))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    ax.set_xlabel('Layer', fontsize=18)
    ax.set_ylabel('Neuron Entropy', fontsize=18)
    plt.savefig('Entropy_on_train_set_10', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(16, 8))
    ax = sns.histplot(val_df, stat='probability', bins=60, x='layer', y='value', hue='dataset', multiple='layer',
                      pthresh=0.03, cbar=True, weights='weight', legend=False, cbar_kws=dict(shrink=.75))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    ax.set_xlabel('Layer', fontsize=18)
    ax.set_ylabel('Neuron Entropy', fontsize=18)
    plt.savefig('Entropy_on_val_set_10', bbox_inches='tight')
    print(1)

    train_df, val_df = [], []
    run = api.runs(args.project, filters={"display_name": 'split_1.0'})[0]
    for i in range(0, 15):
        tag = 'trainset/entropy_dist_{}'.format(str(i).zfill(2))
        dist = run.history(keys=[tag])[tag][1]
        df = {'value': dist, 'layer': i, 'dataset': 'train', 'exp': run.name, 'weight': 1 / len(dist)}
        train_df.append(pd.DataFrame(df))

        tag = 'entropy/layer/{}'.format(str(i).zfill(2))
        dist = run.history(keys=[tag])[tag][1]
        df = {'value': dist, 'layer': i, 'dataset': 'val', 'exp': run.name, 'weight': 1 / len(dist)}
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
    for i in range(0, 15):
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

    for i in range(10, 15):
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
