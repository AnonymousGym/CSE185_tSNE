import numpy as np
import pandas as pd
import scipy.optimize               
import scipy
import math                       
import os

def calc_pji(D, sigmas):
    M = D.shape[0]          
    pji = np.zeros((M,M))
    for i in range(M):
        for j in range(M):   
            pji[i,j] = np.exp(-D[i,j]**2 / (2 * sigmas[i]**2)) if j != i else 0.0
        Z = np.sum(pji[i])
        pji[i] = pji[i] / Z
    return pji  

def calc_D(X):
    M = X.shape[0]    
    D = np.zeros((M,M))
    for i in range(M):
        for j in range(0,i):
            D[i,j] = np.linalg.norm(X[i] - X[j])
            D[j,i] = D[i,j]
    return D

def calc_perplexity_diff(sigma, i, Di, target_perplexity):  
    M = len(Di)
    pji = np.zeros(M)
    for j in range(M):
        pji[j] = -Di[j]**2 / (2 * sigma**2) 
    pji[i] = -math.inf                    
    Z = scipy.special.logsumexp(pji)
    pji = pji - Z
    pji = np.exp(pji)
    H = 0                 
    for j in range(len(Di)):
        if pji[j] > 0:
            H -= pji[j] * np.log2(pji[j])
    perplexity = 2**H
    return perplexity - target_perplexity

def calc_sigmas(D, target_perplexity):
    M = D.shape[0]
    sigma = np.zeros(M)
    for i in range(M):
        a, b = 1.0, 1.0  
        while calc_perplexity_diff(a, i, D[i], target_perplexity) >= 0: 
            a /= 2 
        while calc_perplexity_diff(b, i, D[i], target_perplexity) <= 0: 
            b *= 2  
        sigma[i] = scipy.optimize.bisect(calc_perplexity_diff, a, b, args=(i, D[i],target_perplexity))
    return sigma

def calc_P(X, target_perplexity):
    D = calc_D(X)    
    sigmas = calc_sigmas(D, target_perplexity)    
    pji = calc_pji(D, sigmas)
    P = (pji + pji.T) / (2 * X.shape[0])
    return P