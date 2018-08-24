.. GPyTorch documentation master file, created by
   sphinx-quickstart on Tue Aug 21 09:04:16 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/cornellius-gp/gpytorch

GPyTorch's documentation
====================================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials:

   examples/01_Simple_GP_Regression/Simple_GP_Regression.ipynb
   examples/02_Simple_GP_Classification/Simple_GP_Classification.ipynb

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Examples:

   examples/README.md
   examples/**/index

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   models
   likelihoods
   kernels
   means
   marginal_log_likelihoods
   distributions
   priors

.. toctree::
   :maxdepth: 1
   :caption: Settings and Beta Features

   settings
   beta_features

.. toctree::
   :maxdepth: 1
   :caption: Advanced Package Reference

   module
   lazy
   functions
   variational
   utils



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
