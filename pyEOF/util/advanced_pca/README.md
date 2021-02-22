# Advanced Priniciple Component Analysis

### Table of Contents
1. [Project Motivation](#motivation)
2. [Usage](#usage)
4. [Installation](#installation)
3. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

Researchers use Principle Component Analysis (PCA) intending to summarize features, identify structure in data or reduce the number of features. The interpretation of principal components is challenging in most of the cases due to the high amount of cross-loadings (one feature having significant weight across many principal components). Different types of matrix rotations are used to minimize cross-loadings and make factor interpretation easier.

The `custom_PCA` class is the child of `sklearn.decomposition.PCA` and uses varimax rotation and enables dimensionality reduction in complex pipelines with the modified `transform` method.

`custom_PCA` class implements:
 - __varimax rotation__ for better interpretation of principal components
 - dimensionality reduction based on siginificant __feature communalities__ > 0.5
 - dimensionality reduction based on __feature weights significance__ calculated based on sample size
 - __surrogate feature selection__ - only features with maximum laoding are selected instead of principal components

## Usage <a name="usage"></a>

### Example of using varimax rotation:
```
# 3rd party imports
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from advanced_pca import CustomPCA

# load dataset
dataset = datasets.load_diabetes()
X_std = StandardScaler().fit_transform(dataset.data)

# fit pca objects with and without rotation with 5 principal components
standard_pca5 = CustomPCA(n_components=5).fit(X_std)
varimax_pca5 = CustomPCA(n_components=5, rotation='varimax').fit(X_std)

# display factor matrices and number of cross loadings
print('Factor matrix:\n', standard_pca5.components_.round(1))
print(' Number of cross-loadings:', standard_pca5.count_cross_loadings())
print('\nRotated factor matrix:\n', varimax_pca5.components_.round(1))
print(' Number of cross_loadings:', varimax_pca5.count_cross_loadings()
```
```
Factor matrix:
 [[ 0.2  0.2  0.3  0.3  0.3  0.4 -0.3  0.4  0.4  0.3]
 [ 0.  -0.4 -0.2 -0.1  0.6  0.5  0.5 -0.1 -0.  -0.1]
 [ 0.5 -0.1  0.2  0.5 -0.1 -0.3  0.4 -0.4  0.1  0.3]
 [-0.4 -0.7  0.5 -0.  -0.1 -0.2 -0.1  0.   0.3  0.1]
 [-0.7  0.4  0.1  0.5  0.1  0.1  0.2 -0.1 -0.2  0. ]]
 Number of cross-loadings: 20

Rotated factor matrix:
 [[ 0.1  0.   0.1  0.1  0.6  0.6  0.   0.4  0.2  0.1]
 [ 0.1  0.1  0.5  0.6  0.2  0.1 -0.1  0.2  0.4  0.4]
 [ 0.   0.2  0.3 -0.1 -0.   0.1 -0.7  0.5  0.3  0.2]
 [-0.1 -0.9  0.1 -0.3  0.1 -0.1  0.2 -0.2  0.1 -0.1]
 [-0.9 -0.1  0.1 -0.1 -0.1 -0.1 -0.  -0.1 -0.2 -0.2]]
 Number of cross_loadings: 13
```

### Example of dimensionality reduction based on features' weights and communalities significance:
```
# fit pca objects with option selecting only significant features
significant_pca5 = (CustomPCA(n_components=5, feature_selection='significant')
                    .fit(X_std))

# print selected features based on weights and communalities significance
print('Communalities:\n', significant_pca5.communalities_)
print('\nSelected Features:\n',
      np.asarray(dataset.feature_names)[significant_pca5.get_support()])

# execute dimensionality reduction and pring dataset shapes
print('\nOriginal dataset shape:', X_std.shape)
print('Reduced dataset shape:', significant_pca5.transform(X_std).shape)
```
```
Communalities:
 [0.93669362 0.79747464 0.4109572  0.59415803 0.47225155 0.44619639
 0.55086939 0.35416151 0.24100886 0.1962288 ]

Selected Features:
 ['age' 'sex' 'bp' 's3']

Original dataset shape: (442, 10)
Reduced dataset shape: (442, 4)
```

### Example of selection method of surrogate features:
```
# fit pca objects with option selecting only surrogate features
surrogate_pca = (CustomPCA(rotation='varimax', feature_selection='surrogate')
                 .fit(X_std))

# print factor matrix
print('Factor matrix:\n', surrogate_pca.components_.round(1))
print('\nSelected Features:\n',
      np.asarray(dataset.feature_names)[surrogate_pca.get_support()])

# execute dimensionality reduction and pring dataset shapes
print('\nOriginal dataset shape:', X_std.shape)
print('Reduced dataset shape:', surrogate_pca.transform(X_std).shape)
```
```
Factor matrix:
 [[ 0.1  0.   0.1  0.1  0.6  0.7  0.   0.3  0.2  0.1]
 [ 0.   0.2  0.2  0.  -0.1  0.2 -0.7  0.6  0.2  0.1]
 [ 0.1  0.   0.2  0.1  0.3 -0.  -0.1  0.3  0.9  0.2]
 [-0.1 -1.  -0.  -0.1  0.  -0.1  0.2 -0.1 -0.  -0.1]
 [-1.  -0.1 -0.1 -0.2 -0.1 -0.1  0.  -0.1 -0.1 -0.1]
 [ 0.1  0.1  0.2  0.9  0.1  0.  -0.   0.1  0.2  0.2]
 [ 0.1  0.   0.9  0.2  0.1  0.1 -0.2  0.1  0.2  0.2]
 [ 0.1  0.1  0.1  0.2  0.1  0.1 -0.1  0.2  0.2  0.9]
 [ 0.   0.   0.   0.   0.1 -0.1  0.2  1.   0.   0. ]
 [ 0.  -0.   0.   0.   0.8 -0.7  0.   0.   0.   0. ]]

Selected Features:
 ['bmi' 'bp' 's1' 's2' 's3' 's4' 's5' 's6']

Original dataset shape: (442, 10)
Reduced dataset shape: (442, 8)
```
## Installation <a name="installation"></a>

There are several necessary 3rd party libraries beyond the Anaconda distribution of Python which needs to be installed and imported to run code. These are:
 - [rpy2](https://pypi.org/project/rpy2/) Python interface to the R language used to calculate the varimax rotation

```
pip install advanced-pca
```

## File Descriptions <a name="files"></a>

There are additional files:
 - `custom_pca.py` advanced principle component analysis class definition
 - `licence.txt` see MIT lincence to follow
 - `setup.cfg` and `setup.py` used for creating PyPi package

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to [Joseph F. Hair Jr, William C. Black, Barry J. Babin, Rolph E. Anderson](https://www.amazon.com/Multivariate-Data-Analysis-Joseph-Hair/dp/0138132631).
The ones using projects shall follow [MIT lincence](https://github.com/alfredsasko/advanced-principle-component-analysis/blob/master/advanced_pca/license.txt)
