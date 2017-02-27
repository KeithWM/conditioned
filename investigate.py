import pandas
from matplotlib import pyplot as plt

dfs = ()
names = ()
names+= ('taylor',)
# names+= ('int',)
for name in names:
    # dfs+=(pandas.read_csv('df_{}.csv'.format(name), index_col=0),)
    df = pandas.read_csv('df_{}.csv'.format(name), index_col=0)
# df = pandas.concat(dfs, axis=0)

    int_keys = ('N',)
    float_keys = ('beta', 'gamma', 'upsilon', 'T')
    keys = int_keys + float_keys
    # keys = keys[:3]

    # K = len(keys)
    # if K%2 == 0:
    #     fig, axs = plt.subplots(K/2, K-1)
    # else:
    #     fig, axs = plt.subplots((K-1)/2, K)
    # axs = axs.flatten()
    #
    # df['mean_var_R_disc'] = df['mean_var_R'] - 1.25
    #
    # k=0
    # for i, key1 in enumerate(keys[:-1]):
    #     for key2 in keys[i+1:]:
    #         df.plot.scatter(x=key1, y=key2, c='mean_var_R_disc', s=100, logx=True, logy=True, ax=axs[k], colormap='BrBG')
    #         k+= 1

    keys = keys + ('tau',)
    K = len(keys)
    fig, axs = plt.subplots(1, K)
    axs = axs.flatten()

    df['mean_var_R_disc'] = abs(df['mean_var_R'] - 1.25)
    df['tau'] = df['T']/df['N']

    k=0
    for i, key in enumerate(keys):
        df.plot.scatter(x=key, y='mean_var_R', s=100, logx=True, logy=False, ax=axs[k])
        k+= 1

    fig, ax = plt.subplots(1, 1)
    df.plot.scatter(x='tau', y='upsilon', c='mean_var_R', s=100, logx=True, logy=True, ax=ax)
plt.show()