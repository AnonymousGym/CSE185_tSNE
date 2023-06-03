import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize                # scipy minimizer
import scipy
import math                          
from scipy.special import logsumexp
from sklearn.manifold import TSNE    # test benchmark

def calc_pji(D, sigmas):
    M   = D.shape[0]           
    pji = np.zeros((M,M))
    for i in range(M):
        for j in range(M):   
            pji[i,j] = np.exp(-D[i,j]**2 / (2 * sigmas[i]**2)) if j != i else 0.0
        Z = np.sum(pji[i])
        pji[i] = pji[i] / Z
    return pji  

def calc_D(X):
    M = X.shape[0]
    D = np.zeros((M, M))
    indices = np.triu_indices(M, k=1)  # Get the upper triangular indices of D
    
    differences = X[indices[0]] - X[indices[1]]  # Calculate differences between row pairs
    distances = np.linalg.norm(differences, axis=1)  # Calculate pairwise distances
    
    D[indices] = distances  # Assign distances to the upper triangular part of D
    D = D + D.T  # Make D symmetric by adding its transpose
    
    return D

def calc_perplexity_diff(sigmai, i, Di, target_perplexity):  
    M      = len(Di)
    pji    = np.zeros(M)
    for j in range(M):
        pji[j] = -Di[j]**2 / (2 * sigmai**2) 
    pji[i] = -math.inf                       
    Z      = logsumexp(pji)
    pji    = pji - Z
    pji    = np.exp(pji)
    H = 0                    
    for j in range(len(Di)):
        if pji[j] > 0:
            H -= pji[j] * np.log2(pji[j])

    perplexity = 2**H
    return perplexity - target_perplexity

def calc_sigmas(D, target_perplexity):
    M     = D.shape[0]
    sigma = np.zeros(M)
    for i in range(M):
        a, b = 1.0, 1.0   # Start testing a,b at 1.0
        while calc_perplexity_diff(a, i, D[i], target_perplexity) >= 0: a /= 2    # Move a in 0.5x steps until f(a) < 0
        while calc_perplexity_diff(b, i, D[i], target_perplexity) <= 0: b *= 2    #  ... b in 2x steps until f(a) > 0
        sigma[i] = scipy.optimize.bisect(calc_perplexity_diff, a, b, args=(i, D[i],target_perplexity))
    return sigma

def calc_P(X, target_perplexity):
    print("begin calc D")
    D      = calc_D(X)
    print("after calc D")
    print("begin calc sigma")
    sigmas = calc_sigmas(D, target_perplexity)
    print("after calc sigma")
    print("begin cal pji")
    pji    = calc_pji(D, sigmas)
    print("after cal pji")
    P      = (pji + pji.T) / (2 * X.shape[0])
    return P

def calc_Q(Y):
    M = Y.shape[0]
    Q = np.zeros((M, M))
    Z = 0.0
    dist_matrix = np.linalg.norm(Y[:, np.newaxis] - Y, axis=2)
    similarity_matrix = 1.0 / (1 + dist_matrix ** 2)
    np.fill_diagonal(similarity_matrix, 0)
    Q = similarity_matrix
    Z = np.sum(Q)
    Q /= Z
    return Q

def KL_dist(Y, P):
    M      = P.shape[0]
    L      = len(Y) // M
    Y      = np.reshape(Y, (M,L))    
    Q      = calc_Q(Y)
    kldist = 0.
    grad   = np.zeros((M, L))       

    for i in range(M):
        for j in range(M):
            if P[i,j] > 0:
                kldist += P[i,j] * np.log( P[i,j] / Q[i,j] )

    for i in range(M):
        for j in range(M):
            grad[i] += (P[i,j] - Q[i,j]) * (Y[i] - Y[j]) * (1.0 / (1 + np.linalg.norm( Y[i] - Y[j] )**2))
        grad[i] *= 4.0

    return kldist, grad.flatten()  

def my_tsne(X, target_perplexity):
    print("begin calc P")
    P = calc_P(X, target_perplexity)
    print("after calc P")
    Y = np.random.normal(0., 1e-4, (X.shape[0], 2))   
    result = scipy.optimize.minimize(KL_dist, Y.flatten(), args=(P), jac=True)
    Y = result.x.reshape(X.shape[0], 2)
    print (result.success)
    print (result.nit)
    return Y, result.fun  

Color = [ 'xkcd:red',    'xkcd:green',  'xkcd:yellow',  'xkcd:blue',
          'xkcd:orange', 'xkcd:purple', 'xkcd:cyan',    'xkcd:magenta',
          'xkcd:lime',   'xkcd:pink',   'xkcd:teal',    'xkcd:lavender',
          'xkcd:brown',  'xkcd:maroon', 'xkcd:olive',   'xkcd:navy' ]
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

##
## small_tsne use small test dataset, the example is small.txt
##
def small_tsne(infile):
    df = pd.read_table(infile)   
    C = list(df['type'])    
    df  = df.drop('type', axis=1)
    X   = np.log2( 1 + df.values) 
    M,N = X.shape
    X2  = pca(X)
    D  = calc_D(X2)
    w          = 0   # which sigma_i to look at
    nsamples   = 100
    test_sigma = np.linspace(0.1, 4, num=nsamples)
    pdiff      = [ calc_perplexity_diff(test_sigma[i], w, D[w], 0.) for i in range(nsamples) ]
    target_perplexity = 5.0
    Y, KLdist = my_tsne(X2, target_perplexity)
    print(KLdist)
    fig3, ax3 = plt.subplots(1,1)
    for i in range(M):
        ax3.plot(Y[i,0], Y[i,1], 'o', markersize=4, mfc='w', mec=Color[C[i]])
    plt.show()

##
## the large_tsne use large (real) dataset, the workflow is in README file.
##
def large_tsne(v,c):
    D  = calc_D(v)
    w          = 0   # which sigma_i to look at
    nsamples   = 10000
    test_sigma = np.linspace(0.1, 4, num=nsamples)
    pdiff      = [ calc_perplexity_diff(test_sigma[i], w, D[w], 0.) for i in range(nsamples) ]
    target_perplexity = 2.0
    Y, KLdist = my_tsne(v, target_perplexity)
    fig3, ax3 = plt.subplots(1,1)
    for i in range(X.shape[0]):
        ax3.plot(Y[i,0], Y[i,1], 'o', markersize=4, mfc='w',mec=Color[int(c[i])])
    plt.show() 