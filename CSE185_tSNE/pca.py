import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize                # the SciPy function minimizers
import scipy
import math                          # I'm going to use math.inf
from scipy.special import logsumexp

import scanpy as sc
import anndata as ad
import os

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white')


def pca(x, no_dims = 50):
    (n, d) = x.shape
    x = x - np.tile(np.mean(x, 0), (n, 1))
    print("minus!")
    xd = np.dot(x.T,x)
    print("dot!")
    #l, M = np.linalg.eig(xd)
    l,M = scipy.sparse.linalg.eigs(xd)
    print("eig!")
    #l, M = np.linalg.eig(np.dot(x.T, x))
    y = np.dot(x, M[:,0:no_dims])
    print("dot!")
    return y

def read_data(infile):
    df = pd.read_table(infile)    # assume a well-formatted Pandas table, tab-delimited, col headers
    ctype = list(df['type'])      # pull column 'type' out, the cell type labels
    df  = df.drop('type', axis=1)  # delete that column, leaving just the count data columns
    X   = np.log2( 1 + df.values)  # convert to log2(counts + 1). The +1 is arbitrary to avoid log(0)
    M,N = X.shape
    return X, ctype, M, N

def run_pca_test():
    Xs, Cs, Ms, Ns = read_data("small.txt")
    Color = [ 'xkcd:red',    'xkcd:green',  'xkcd:yellow',  'xkcd:blue',
          'xkcd:orange', 'xkcd:purple', 'xkcd:cyan',    'xkcd:magenta',
          'xkcd:lime',   'xkcd:pink',   'xkcd:teal',    'xkcd:lavender',
          'xkcd:brown',  'xkcd:maroon', 'xkcd:olive',   'xkcd:navy' ]
    Xs2 = pca(Xs)
    fig1, (ax1A, ax1B) = plt.subplots(1,2, figsize=(9,4))  # a figure with two panels, side by side
    for i in range(Ms):
        ax1A.plot(Xs2[i,0], Xs2[i,1], 'o', markersize=4, mfc='w', mec=Color[Cs[i]%16])
    plt.show()

def run_pca(X):
    return pca(X)