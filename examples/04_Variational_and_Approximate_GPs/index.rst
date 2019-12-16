
Variational and Approximate GPs
================================

Variational and approximate Gaussian processes are used in a variety of cases:

- When the GP likelihood is non-Gaussian (e.g. for classification).
- To scale up GP regression (by using stochastic optimization).
- To use GPs as part of larger probablistic models.

With GPyTorch it is possible to implement various types approximate GP models.
All approximate models consist of the following 3 composible objects:

- :obj:`VariationalDistribution`, which define the form of the approximate inducing value
  posterior :math:`q(\mathbf u)`.
- :obj:`VarationalStrategies`, which define how to compute :math:`q(\mathbf f(\mathbf X))` from
  :math:`q(\mathbf u)`.
- :obj:`~gpytorch.mlls._ApproximateMarginalLogLikelihood`, which defines the objective function
  to learn the approximate posterior (e.g. variational ELBO).

The variational documentation has more information on how to use these objects.
Here we provide some examples which highlight some of the common use cases:

- **Large-scale regression** (when exact methods are too memory intensive): see the `stochastic variational regression example`_.
- **Alternative optimization objectives** for the GP's predictive distribution: see the `approximate GP objective functions notebook`_.
  This example compares and contrasts the variational ELBO with the predictive log likelihood of Jankowiak et al., 2019.
- **Classification**: see the `non-Gaussian likelihood notebook`_.
- **Multi-output variational GPs** (when exact methods are too memory intensive): see the `variational GPs with multiple outputs example`_.
- **Uncertain inputs**: see the `GPs with uncertain inputs example`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   SVGP_Regression_CUDA.ipynb
   Approximate_GP_Objective_Functions.ipynb
   Non_Gaussian_Likelihoods.ipynb
   SVGP_Multitask_GP_Regression.ipynb
   GP_Regression_with_Uncertain_Inputs.ipynb

.. _stochastic variational regression example:
  ./SVGP_Regression_CUDA.ipynb

.. _approximate GP objective functions notebook:
  ./Approximate_GP_Objective_Functions.ipynb

.. _non-Gaussian likelihood notebook:
  ./Non_Gaussian_Likelihoods.ipynb

.. _variational GPs with multiple outputs example:
  ./SVGP_Multitask_GP_Regression.ipynb

.. _GPs with uncertain inputs example:
  ./GP_Regression_with_Uncertain_Inputs.ipynb
