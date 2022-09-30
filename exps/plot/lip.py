import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
sns.set_theme(style="darkgrid")
from config import *

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': large,
          'figure.figsize': (12, 8),
          'axes.labelsize': large,
          'xtick.labelsize': large,
          'ytick.labelsize': large,

          'figure.titlesize': large}
plt.rcParams.update(params)
fig, ax =plt.subplots()
np_file_0 = os.path.join(MODEL_PATH, 'exp', 'lip.npy')
flt0 = np.load(np_file_0)

# df = pd.DataFrame(data=d, columns=['STD', r'$\sigma$-0.05', r'$\sigma$-0.10', r'$\sigma$-0.25'])
# sns.violinplot(data=df, split=True, ax=ax)


fig, ax = plt.subplots()
ax2 = ax.twinx()
linecolor = ['r', 'g', 'b', 'y']
for i in range(4):
    for j in range(2):
        if j ==0:
            linetype = '-'
            d = np.log(flt0[i, :, j, :].mean(axis=1))
            ax.plot(np.arange(1, 17), d, linecolor[i] + linetype)
        else:
            linetype = '--'
            d = flt0[i, :, j, :].mean(axis=1)
            ax2.plot(np.arange(1, 17), d, linecolor[i]+linetype)
# for i in range(4):
#     data = pd.DataFrame(
#         {'0.1': flt0[i].mean(axis=1), '0.25': flt1[i].mean(axis=1)})
#
#     sns.violinplot(data=data, ax=ax[i])
#     # ax[i].violinplot(data=data, split=True)
#     ax[i].set_ylim(0.6, 0.97)
# ax.set_xticks(['STD', '$\sigma$-0.10', '$\sigma$-0.05', '$\sigma$-0.05'])
ax.set_ylabel('Mean($L_2$)')
ax2.set_ylabel('Variance($L_2$)')
# ax.set_ylabel2('#fixed neuron / # neurons')
ax.set_xlabel('Perturbation Size')
ax.legend(['STD', '$\sigma=0.05$', '$\sigma=0.10$', '$\sigma=0.25$'])
plt.show()
