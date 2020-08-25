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

   examples/01_Exact_GPs/Simple_GP_Regression.ipynb

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Examples:

   examples/**/index

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   models
   likelihoods
   kernels
   means
   marginal_log_likelihoods
   constraints
   distributions
   priors
   variational
   optim

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
   utils



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Research references
======================

* Gardner, Jacob R., Geoff Pleiss, David Bindel, Kilian Q. Weinberger, and Andrew Gordon Wilson. " GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration." In NeurIPS (2018).
* Pleiss, Geoff, Jacob R. Gardner, Kilian Q. Weinberger, and Andrew Gordon Wilson. "Constant-Time Predictive Distributions for Gaussian Processes." In ICML (2018).
* Gardner, Jacob R., Geoff Pleiss, Ruihan Wu, Kilian Q. Weinberger, and Andrew Gordon Wilson. "Product Kernel Interpolation for Scalable Gaussian Processes." In AISTATS (2018).
* Wilson, Andrew G., Zhiting Hu, Ruslan R. Salakhutdinov, and Eric P. Xing. "Stochastic variational deep kernel learning." In NeurIPS (2016).
* Wilson, Andrew, and Hannes Nickisch. "Kernel interpolation for scalable structured Gaussian processes (KISS-GP)." In ICML (2015).
* Hensman, James, Alexander G. de G. Matthews, and Zoubin Ghahramani. "Scalable variational Gaussian process classification." In AISTATS (2015).
