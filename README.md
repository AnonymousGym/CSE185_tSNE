# CSE185-DRV
Version 2.7.1 (Fixed pca and tsne)

DRV stands for Dimensional Reduction and Visualization, just like its name, this tool aims to work for the biological datasets that usually have multiple variables with relatively high dimension that people cannot describe any rational tendency by just visualize by eye or use traditional data analysis methods. DRV reduce datasets' dimensions to the most visualizable state and presents only the top significant features in order to describe the trends of the datasets. In order to maintain the data quality, our tool also tries to minimize the loss of relevant information, so the efficiency and accuracy is guaranteed. The final result would be a 2-dimensional graph that illustrate the relationship between the variables.

We have completed 3 major functions: pca, small_tsne and large_tsne. The usage can be found below.
# Installing
Here is the pip install command to install our package, the current version is 2.7.1:
```
pip install -i https://test.pypi.org/simple/ CSE185-tSNE
```
Then, you can import CSE185_tSNE in your python script.

You should use:
```
from CSE185_tSNE import *
```
# Using commands
There are some working commands for Version 2.7.1:

1. Use ```pca.run_pca(X)``` function: You should input a count matrix, and it will return the calculated matrix to you. For detailed usage, you can refer to test.py above. Basically, you should follow the scanpy analysis pipeline, and run our run_pca function with adata.X, which is the count matrix of anndata objects. After running run_pca, you should use leiden method to cluster the data, and then visualize it. You will be able to see the pca plot of your adata object.

2. Use ```small_tsne(infile)``` function: You should download small.txt, and use it as example infile. If you want to use other infile, you can refer to its format. Prepare to label the clusters first! You can look at Small_Example.ipynb to find usage examples.

3. Use ```large_tsne(v,c)``` function: You should follow the scanpy analysis pipeline and use it. You can find example code in our github. However, our tsne is hard to be used on large count matrices. So we don't expect you to use this function well. You can refer to Large_Example.ipynb to look at the details.

After using these functions, you should use various colors to generate your plot! Make sure the number of different colors is enough for your number of clusters. You can update the colors in the ```Colors``` list.
# Benchmarking
We did the benchmarking with functions timed. All results can be found in the Benchmark.ipynb.
# Package Maintaining
Here is the command to update python packages uploaded to TestPyPI.
```
python3 -m pip install --user --upgrade setuptools wheel
python3 -m pip install --user --upgrade twine
python3 setup.py sdist bdist_wheel
ls dist
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
