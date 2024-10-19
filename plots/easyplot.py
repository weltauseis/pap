#!/usr/bin/env python3
from graphTools import *
import sys

assert sys.version_info >= (3,8)

args = parseArguments(sys.argv)
df = getDataFrame(args)

# see customizing-with-matplotlibrc-files
# https: // matplotlib.org/tutorials/introductory/customizing.html

sns.set(style="darkgrid", rc={'text.usetex': False,
                              'legend.handletextpad': 0,
                              'figure.titlesize': 'medium'})

# plt.style.use(['dark_background'])


# Selection des lignes :
# df = df[(-df.threads.isin([8])) & (df.kernel.isin(['mandel']))].reset_index(drop = True)


# Creation du graphe :
fig = easyPlotDataFrame(df=df, args=args)

savePlot(fig)
