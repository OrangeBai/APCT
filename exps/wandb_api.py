import re

import matplotlib.pyplot as plt
import wandb

WANDB_DIR = "/home/orange/Main/Experiment/ICLR/cifar10/cifar10/vgg16_express2/"
api = wandb.Api()
runs = api.runs("orangebai/express2")

import seaborn as sns

sns.set()
sns.set_theme(style="darkgrid")

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': large,
          'figure.figsize': (16, 9),
          'axes.labelsize': large,
          'xtick.labelsize': large,
          'ytick.labelsize': large,

          'figure.titlesize': large}
plt.rcParams.update(params)
# Project is specified by <entity/project-name>
numeric_const_pattern = r"""
     [-+]? # optional sign
     (?:
         (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
         |
         (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
     )
     # followed by optional exponent part if desired
     (?: [Ee] [+-]? \d+ ) ?
     """
rx = re.compile(numeric_const_pattern, re.VERBOSE)

bench_id = ''
for run in runs:
    if run.name == 'batchsize: 128':
        bench_id = run.id
        break

batch_ids = {128: bench_id}
for run in runs:
    print(run.name)
    if 'batchsize' in run.name and 'split' not in run.name:
        batch_size = rx.findall(run.name)[0]
        batch_ids[int(batch_size)] = run.id

# lines = []
# color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# for i in [1, 6,  11]:
#     fig, ax = plt.subplots()
#     entropy_name = 'entropy/layer/{:02}'.format(i)
#     var_name = 'entropy/layer_var/{:02}'.format(i)
#     print(entropy_name)
#     keys = list(batch_ids.keys())
#     keys.sort()
#     for k, c in zip(keys, color):
#         run = api.run("orangebai/express2/{}".format(batch_ids[k]))
#         data = run.history(keys=["step", entropy_name, var_name])
#         x = data['step']
#         y = data[entropy_name]
#         ax.plot(x, y, color=c, label='batch size:' + '{}'.format(k).rjust(3))
#     ax.set_xlabel('Steps')
#     ax.set_ylabel('Float Entropy')
#     # ax.set_title('Layer-{:0d}'.format(i))
#     # plt.legend()# fig.legend()
#     plt.savefig('plot/figs/mean_layer_{0:02d}.png'.format(i), bbox_inches='tight')
#     plt.show()
#
#
# lines = []
# color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# for i in [1, 6,  11]:
#     fig, ax = plt.subplots()
#     entropy_name = 'entropy/layer/{:02}'.format(i)
#     var_name = 'entropy/layer_var/{:02}'.format(i)
#     print(entropy_name)
#     keys = list(batch_ids.keys())
#     keys.sort()
#     for k, c in zip(keys, color):
#         run = api.run("orangebai/express2/{}".format(batch_ids[k]))
#         data = run.history(keys=["step", entropy_name, var_name])
#         x = data['step']
#         y = data[var_name]
#         ax.plot(x, y, color=c, label='batch size:' + '{}'.format(k).rjust(3))
#     ax.set_xlabel('Steps')
#     ax.set_ylabel('Average Entropy')
#     # plt.legend()# fig.legend()
#     # fig.legend(['batch size:' + '{}'.format(k).rjust(3) for k in keys], loc=8)
#     plt.savefig('plot/figs/var_layer_{0:02d}.png'.format(i), bbox_inches='tight')


lines = []
color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
fig, ax = plt.subplots()
keys = list(batch_ids.keys())
keys.sort()
for k, c in zip(keys, color):
    run = api.run("orangebai/express2/{}".format(batch_ids[k]))
    data = run.history(keys=["_step", 'train' + '/top1'])
    x = data['_step']
    y = data['train' + '/top1']
    ax.plot(x, y, color=c, label='batch size:' + '{}'.format(k).rjust(3))
ax.set_xlabel('Steps')
ax.set_ylabel('Average Entropy')
plt.legend()# fig.legend()
# fig.legend(['batch size:' + '{}'.format(k).rjust(3) for k in keys], loc=8)
plt.savefig('plot/figs/{}_acc.png'.format('train'), bbox_inches='tight')


fig, ax = plt.subplots()
for k, c in zip(keys, color):
    run = api.run("orangebai/express2/{}".format(batch_ids[k]))
    data = run.history(keys=["step", 'val' + '/top1'])
    x = data['step']
    y = data['val' + '/top1']
    ax.plot(x, y, color=c, label='batch size:' + '{}'.format(k).rjust(3))
ax.set_xlabel('Steps')
ax.set_ylabel('Average Entropy')
plt.legend()# fig.legend()
# fig.legend(['batch size:' + '{}'.format(k).rjust(3) for k in keys], loc=8)
plt.savefig('plot/figs/{}_acc.png'.format('val'), bbox_inches='tight')