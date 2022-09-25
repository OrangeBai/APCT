import matplotlib.pyplot as plt
import seaborn as sns


large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 9),
          'axes.labelsize': med,
          # 'xtick.labelsize': med,
          # 'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-dark')
sns.set_theme(style="darkgrid")