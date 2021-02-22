About
======================================================================================

|doi| |docs| |GitHub| |binder| |license|

.. |doi| image:: https://zenodo.org/badge/341276703.svg
   :target: https://zenodo.org/badge/latestdoi/341276703

.. |docs| image:: https://readthedocs.org/projects/pyeof/badge/?version=latest
   :target: https://pyeof.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |GitHub| image:: https://img.shields.io/badge/GitHub-pyEOF-brightgreen.svg
   :target: https://github.com/zzheng93/pyEOF

.. |binder| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/zzheng93/pyEOF/HEAD?filepath=docs%2Fnotebooks

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/zzheng93/pyEOF/blob/master/LICENSE

pyEOF is a **Python** package for `EOF and Rotated EOF Analysis <https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/empirical-orthogonal-function-eof-analysis-and-rotated-eof-analysis>`_ . It takes advantage of

- `sklearn.decomposition.PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_ (for EOF)
- `Advanced Priniciple Component Analysis <https://github.com/alfredsasko/advanced-principle-component-analysis>`_ (for varimax rotation //  varimax rotated EOF // REOF)
