Installation
============

Step 1: create an environment::

    $ conda create -n pyEOF python=3.7
    $ conda activate pyEOF
    $ conda install -c conda-forge numpy pandas scipy scikit-learn rpy2

Step 2: install using pip::

    $ pip install pyEOF

(optional) for jupyter notebook tutorial:: 

    $ conda install -c conda-forge numpy pandas scipy scikit-learn rpy2 xarray matplotlib jupyter eofs pooch

(optional) install from source:: 

    $ git clone https://github.com/zzheng93/pyEOF.git
    $ cd pyEOF
    $ python setup.py install
