## This is the testbench for pca method. It tests with a larger dataset. 
## The large dataset is in the folder large_test, which is the sc-Seq data from lab5.
## The outputs a valid pca result plot.

from CSE185_tSNE import *
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import os

sc.settings.verbosity = 3            
sc.settings.set_figure_params(dpi=80, facecolor='white')

import matplotlib.pyplot as plt
import scipy
import math                        
from scipy.special import logsumexp


ds = "GSM5114461_S6_A11"
adata = sc.read_10x_mtx(path="./large_test",prefix=ds+"_", cache=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
X = pd.DataFrame(adata.X)

Color = [ 'xkcd:red',    'xkcd:green',  'xkcd:yellow',  'xkcd:blue',
          'xkcd:orange', 'xkcd:purple', 'xkcd:cyan',    'xkcd:magenta',
          'xkcd:lime',   'xkcd:pink',   'xkcd:teal',    'xkcd:lavender',
          'xkcd:brown',  'xkcd:maroon', 'xkcd:olive',   'xkcd:navy' ]

X2  = pca.run_pca(X)
v = X2.astype(np.float32)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata)
sc.tl.umap(adata)
c = adata.obs.leiden
for i in range(X.shape[0]):
    plt.plot(v[i,0], v[i,1], 'o', markersize=4, mfc='w',mec=Color[int(c[i])])
plt.show()
