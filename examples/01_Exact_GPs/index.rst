Exact GPs (Regression)
========================

Regression with a Gaussian noise model is the cannonical example of Gaussian processes.
These examples will work for small to medium sized datasets (~2,000 data points).
All examples here use exact GP inference.

- `Simple GP Regression`_ is the basic tutorial for regression in GPyTorch.
- `Spectral Mixture Regression`_ extends on the previous example with a more complex kernel.
- `Fully Bayesian GP Regression`_ demonstrates how to perform fully Bayesian inference by sampling the GP hyperparameters
  using NUTS. (This example requires Pyro to be installed).
- `Distributional GP Regression`_ is an example of how to take account of uncertainty in inputs.

.. toctree::
   :maxdepth: 1
   :hidden:

   Simple_GP_Regression.ipynb
   Spectral_Delta_GP_Regression.ipynb
   Spectral_Mixture_GP_Regression.ipynb
   GP_Regression_Fully_Bayesian.ipynb
   GP_Regression_DistributionalKernel.ipynb

.. _Simple GP Regression:
  ./Simple_GP_Regression.ipynb

.. _Spectral Mixture Regression:
  ./Spectral_Mixture_GP_Regression.ipynb

.. _Fully Bayesian GP Regression:
  ./GP_Regression_Fully_Bayesian.ipynb

.. _Distributional GP Regression:
  ./GP_Regression_DistributionalKernel.ipynb
