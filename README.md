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
#1


Here is the command to update python packages uploaded to TestPyPI.
```
python3 -m pip install --user --upgrade setuptools wheel
python3 -m pip install --user --upgrade twine
python3 setup.py sdist bdist_wheel
ls dist
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
