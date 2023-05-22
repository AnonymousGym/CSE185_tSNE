
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize                # the SciPy function minimizers
import math                          # I'm going to use math.inf
from scipy.special import logsumexp
#from sklearn.manifold import TSNE    # scikit-learn's t-SNE implementation

def read_data(infile):
    df = pd.read_table(infile)    # assume a well-formatted Pandas table, tab-delimited, col headers
    ctype = list(df['type'])      # pull column 'type' out, the cell type labels
    df  = df.drop('type', axis=1)  # delete that column, leaving just the count data columns
    X   = np.log2( 1 + df.values)  # convert to log2(counts + 1). The +1 is arbitrary to avoid log(0)
    M,N = X.shape
    return X, ctype, M, N

#Xs, Cs, Ms, Ns = read_data("w13-data-small.tbl")
#X,  C,  M,  N  = read_data("w13-data-large.tbl")
def PCA(X):
    Xc       = X - np.mean(X, axis=0)   # 'center' the data: subtract column means
    U, S, Wt = np.linalg.svd(Xc)        # singular value decomposition
    W        = np.transpose(Wt)
    eigvals  = S*S / Xc.shape[0]     
    X2       = Xc @ W[:,:2]             # projection to 2D PCs
    return X2

def test():
    Xs, Cs, Ms, Ns = read_data("http://mcb112.org/w13/w13-data-small.tbl")
    X,  C,  M,  N  = read_data("http://mcb112.org/w13/w13-data-large.tbl")
    Color = [ 'xkcd:red',    'xkcd:green',  'xkcd:yellow',  'xkcd:blue',
          'xkcd:orange', 'xkcd:purple', 'xkcd:cyan',    'xkcd:magenta',
          'xkcd:lime',   'xkcd:pink',   'xkcd:teal',    'xkcd:lavender',
          'xkcd:brown',  'xkcd:maroon', 'xkcd:olive',   'xkcd:navy' ]
    Xs2 = PCA(Xs)
    X2  = PCA(X)
    fig1, (ax1A, ax1B) = plt.subplots(1,2, figsize=(9,4))  # a figure with two panels, side by side
    for i in range(Ms):
        ax1A.plot(Xs2[i,0], Xs2[i,1], 'o', markersize=4, mfc='w', mec=Color[Cs[i]%16])
    for i in range(M):
        ax1B.plot(X2[i,0], X2[i,1], 'o', markersize=4, mfc='w', mec=Color[C[i]%16])
    plt.show()