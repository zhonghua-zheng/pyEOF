======================================================================================
pyEOF: Empirical Orthogonal Function (EOF) analysis and Rotated EOF analysis in Python
======================================================================================
|docs| |GitHub| |license|

.. |docs| image:: https://readthedocs.org/projects/clhs-py/badge/?version=latest
   :target: https://clhs-py.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |GitHub| image:: https://img.shields.io/badge/GitHub-clhs__py-informational.svg
   :target: https://github.com/zzheng93/pyEOF

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/zzheng93/pyEOF/blob/master/LICENSE

pyEOF is a **Python** package for `EOF and Rotated EOF Analysis <https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/empirical-orthogonal-function-eof-analysis-and-rotated-eof-analysis>`_ . It takes advantage of

- `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_ (for EOF)
- `Advanced Priniciple Component Analysis <https://github.com/alfredsasko/advanced-principle-component-analysis>`_ (for varimax rotation //  varimax rotated EOF // REOF)


Installation
------------

Step 1: create an environment::

    $ conda create -n pyEOF python=3.7
    $ conda activate pyEOF
    $ conda install -c conda-forge numpy pandas scipy scikit-learn rpy2

Step 2: install using pip::

    $ pip install pyEOF

(optional) for jupyter notebook tutorial:: 

    $ conda install -c conda-forge numpy pandas scipy scikit-learn rpy2 xarray matplotlib jupyter eofs

(optional) install from source:: 

    $ git clone https://github.com/zzheng93/pyEOF.git
    $ cd pyEOF
    $ python setup.py install