======================================================================================
pyEOF: Empirical Orthogonal Function (EOF) analysis and Rotated EOF analysis in Python
======================================================================================
|doi| |docs| |GitHub| |binder| |license| |pepy|

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4556050.svg
   :target: https://doi.org/10.5281/zenodo.4556050

.. |docs| image:: https://readthedocs.org/projects/pyeof/badge/?version=latest
   :target: https://pyeof.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |GitHub| image:: https://img.shields.io/badge/GitHub-pyEOF-brightgreen.svg
   :target: https://github.com/zzheng93/pyEOF

.. |binder| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/zzheng93/pyEOF/HEAD?filepath=docs%2Fnotebooks

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/zzheng93/pyEOF/blob/master/LICENSE
   
.. |pepy| image:: https://static.pepy.tech/personalized-badge/pyeof?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads
   :target: https://pepy.tech/project/pyeof

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

How to use it?
--------------
You may reproduce the jupyter notebook example on `Binder <https://mybinder.org/v2/gh/zzheng93/pyEOF/HEAD?filepath=docs%2Fnotebooks>`_.

Please check `online documentation <https://pyeof.readthedocs.io/en/latest/>`_ for more information.

How to ask for help
-------------------
The `GitHub issue tracker <https://github.com/zzheng93/pyEOF/issues>`_ is the primary place for bug reports. 
