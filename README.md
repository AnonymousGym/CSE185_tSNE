# CSE185-DRV
Version 2.5

DRV stands for Dimensional Reduction and Visualization, just like its name, this tool aims to work for the biological datasets that usually have multiple variables with relatively high dimension that people cannot describe any rational tendency by just visualize by eye or use traditional data analysis methods. DRV reduce datasets' dimensions to the most visualizable state and presents only the top significant features in order to describe the trends of the datasets. In order to maintain the data quality, our tool also tries to minimize the loss of relevant information, so the efficiency and accuracy is guaranteed. The final result would be a 2-dimensional graph that illustrate the relationship between the variables.

Now, we have only pca module ready to be used. We have also completed a few tSNE functions and will continue to test on these during the weekend.

Here is the pip install command to install our package:
```
pip install -i https://test.pypi.org/simple/ CSE185-tSNE
```
Then, you can import CSE185_tSNE in your python script.
There are some working commands for Version 2.5:

1. Use pca.run_pca_test() function: You don't need to specify an input. This function runs on the small pre-labeled dataset small.txt. It will generate a pca plot of small.txt. 

2. Use pca.run_pca(X) function: You should input a count matrix, and it will return the calculated matrix to you. For detailed usage, you can refer to test.py above. Basically, you should follow the scanpy analysis pipeline, and run our run_pca function with adata.X, which is the count matrix of anndata objects. After running run_pca, you should use leiden method to cluster the data, and then visualize it. You will be able to see the pca plot of your adata object.

# To be brief, you can take a look at test.py. If you install our tSNE tool and create a python file, copy the script from test.py and run it, you will be able to generate the pca plot.

A simpler version of code instruction is below:
```
X = pd.DataFrame(adata.X)
X2  = pca.run_pca(X)
v = X2.astype(np.float32)
```
Obtain count matrix and run pca. You will have to cast the result to float type.
```
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata)
sc.tl.umap(adata)
c = adata.obs.leiden
```
Find out the clusters in the data, so that we can label the pca plot.
```
Color = [ 'xkcd:red',    'xkcd:green',  'xkcd:yellow',  'xkcd:blue',
          'xkcd:orange', 'xkcd:purple', 'xkcd:cyan',    'xkcd:magenta',
          'xkcd:lime',   'xkcd:pink',   'xkcd:teal',    'xkcd:lavender',
          'xkcd:brown',  'xkcd:maroon', 'xkcd:olive',   'xkcd:navy']
for i in range(X.shape[0]):
    plt.plot(v[i,0], v[i,1], 'o', markersize=4, mfc='w',mec=Color[int(c[i])])
plt.show()
```
Use various colors to generate your plot! Make sure the number of different colors is enough for your number of clusters.

Here is the command to update python packages uploaded to TestPyPI.
```
python3 -m pip install --user --upgrade setuptools wheel
python3 -m pip install --user --upgrade twine
python3 setup.py sdist bdist_wheel
ls dist
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
