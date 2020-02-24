Advanced Usage
===============================================

Here are some examples highlighting GPyTorch's more advanced features.

Batch GPs
-----------

GPyTorch makes it possible to train/perform inference with a batch of Gaussian processes in parallel.
This can be useful for a number of applications:

 - Modeling a function with multiple (independent) outputs
 - Performing efficient cross-validation
 - Parallel acquisition function sampling for Bayesian optimization
 - And more!

Here we highlight a number of common batch GP scenarios and how to construct them in GPyTorch.

- **Multi-output functions (with independent outputs).** Batch GPs are extremely efficient at modelling multi-output functions, when each of the output functions
  are **independent**. See the `Batch Independent Multioutput GP`_ example for more details.

- **For cross validation**, or for some BayesOpt applications, it may make sense to evaluate the GP on different batches of test data.
  This can be accomplished by using a standard (non-batch) GP model.
  At test time, feeding a `b x n x d` tensor into the model will then return `b` batches of `n` test points.
  See the `Batch Mode Regression`_ example for more details.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   Simple_Batch_Mode_GP_Regression.ipynb


GPs with Derivatives
----------------------

Derivative information can be used by GPs to accelerate Bayesian optimization.
See the `1D derivatives GP example`_ or the `2D derivatives GP example`_ for examples on using GPs with derivative information.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   Simple_GP_Regression_Derivative_Information_1d.ipynb
   Simple_GP_Regression_Derivative_Information_2d.ipynb

.. _Batch Independent Multioutput GP:
  Batch_Independent_Multioutput_GP.ipynb

.. _Batch Mode Regression:
  Simple_Batch_Mode_GP_Regression.ipynb

.. _1D derivatives GP example:
  Batch_Independent_Multioutput_GP.ipynb

.. _2D derivatives GP example:
  Simple_Batch_Mode_GP_Regression.ipynb:


Converting Models to TorchScript
----------------------------------

In order to deploy GPs in production code, it can be desirable to avoid using PyTorch directly for performance reasons.
Fortunarely, PyTorch offers a mechanism caled TorchScript to aid in this. In these example notebooks, we'll demonstrate
how to convert both an exact GP and a variational GP to a ScriptModule that can then be used for example in LibTorch.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   TorchScript_Exact_Models.ipynb
   TorchScript_Variational_Models.ipynb
