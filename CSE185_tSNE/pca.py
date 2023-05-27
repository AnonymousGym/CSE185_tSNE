import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize             
import scipy
import math                        
import os

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

def run_pca_test():
    df = pd.read_table("small.txt")   
    Cs = list(df['type'])   
    df  = df.drop('type', axis=1) 
    Xs   = np.log2( 1 + df.values)  
    Ms,Ns = Xs.shape
    Color = [ 'xkcd:red',    'xkcd:green',  'xkcd:yellow',  'xkcd:blue',
          'xkcd:orange', 'xkcd:purple', 'xkcd:cyan',    'xkcd:magenta',
          'xkcd:brown',  'xkcd:maroon', 'xkcd:olive',   'xkcd:navy' ,
          'xkcd:lime',   'xkcd:pink',   'xkcd:teal',    'xkcd:lavender']
    Xs2 = pca(Xs)
    for i in range(Ms):
        plt.plot(Xs2[i,0], Xs2[i,1], 'o', markersize=4, mfc='w', mec=Color[Cs[i]%16])
    plt.show()

def run_pca(X):
    return pca(X)