
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

def calc_pji(D, sigmas):
    M   = D.shape[0]           # M = number of rows, samples
    pji = np.zeros((M,M))
    for i in range(M):
        for j in range(M):     # it's not symmetric, because of sigma_i
            pji[i,j] = np.exp(-D[i,j]**2 / (2 * sigmas[i]**2)) if j != i else 0.0
        Z = np.sum(pji[i])
        pji[i] = pji[i] / Z
    return pji    

def calc_D(X):
    M = X.shape[0]       # ncells
    D = np.zeros((M,M))
    for i in range(M):
        for j in range(0,i):
            D[i,j] = np.linalg.norm(X[i] - X[j])
            D[j,i] = D[i,j]
    return D

def calc_perplexity_diff(sigmai, i, Di, target_perplexity):   # <Di> is one row of the distance matrix. I need to know i to set p_ii = 0
    M      = len(Di)
    pji    = np.zeros(M)
    for j in range(M):
        pji[j] = -Di[j]**2 / (2 * sigmai**2)  # doing the calculation in log space to avoid nans and underflows
    pji[i] = -math.inf                        # which means, to get p_ii = 0, log p_ii = -inf
    Z      = logsumexp(pji)
    pji    = pji - Z
    pji    = np.exp(pji)

    H = 0                     # the Shannon entropy calculation for the p_j|i given this sigma_i value
    for j in range(len(Di)):
        if pji[j] > 0:
            H -= pji[j] * np.log2(pji[j])

    perplexity = 2**H
    return perplexity - target_perplexity

def test2():
    Xs, Cs, Ms, Ns = read_data("http://mcb112.org/w13/w13-data-small.tbl")
    X,  C,  M,  N  = read_data("http://mcb112.org/w13/w13-data-large.tbl")
    Color = [ 'xkcd:red',    'xkcd:green',  'xkcd:yellow',  'xkcd:blue',
          'xkcd:orange', 'xkcd:purple', 'xkcd:cyan',    'xkcd:magenta',
          'xkcd:lime',   'xkcd:pink',   'xkcd:teal',    'xkcd:lavender',
          'xkcd:brown',  'xkcd:maroon', 'xkcd:olive',   'xkcd:navy' ]
    Xs2 = PCA(Xs)
    X2  = PCA(X)
    Ds = calc_D(Xs)
    D  = calc_D(X)
    w          = 0   # which sigma_i to look at
    nsamples   = 100
    test_sigma = np.linspace(0.1, 4, num=nsamples)
    pdiff      = [ calc_perplexity_diff(test_sigma[i], w, D[w], 0.) for i in range(nsamples) ]

    fig2, ax2  = plt.subplots(1,1, figsize=(6,3))
    ax2.plot(test_sigma, pdiff)
    ax2.set_xlabel('$\sigma_i$')
    ax2.set_ylabel('perplexity')
    plt.show()