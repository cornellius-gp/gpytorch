Pyro Integration
===================

GPyTorch can optionally work with the Pyro probablistic programming language.
This makes it possible to use Pyro's advanced inference algorithms, or to incorporate GPs as part of larger probablistic models.
GPyTorch offers two ways of integrating with Pyro:

High-level Pyro Interface (for predictive models)
--------------------------------------------------

The high-level interface provides a simple wrapper around :obj:`~gpytorch.models.ApproximateGP` that makes it
possible to use Pyro's inference tools with GPyTorch models.
It is best designed for:

- Developing models that will be used for predictive tasks
- GPs with likelihoods that have additional latent variables

The `Pyro + GPyTorch High-Level Introduction`_ gives an overview of the high-level interface.
For a more in-depth example that shows off the power of the integration, see the `Clustered Multitask GP Example`_.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   Pyro_GPyTorch_High_Level.ipynb
   Clustered_Multitask_GP_Regression.ipynb


Low-level Pyro Interface (for latent function inference)
----------------------------------------------------------

The low-level interface simply provides tools to compute GP latent functions, and requires users to write their own :meth:`model` and :meth:`guide` functions.
It is best designed for:

- Performing inference on probabilistic models that involve GPs
- Models with complicated likelihoods

The `Pyro + GPyTorch Low-Level Introduction`_ gives an overview of the low-level interface.
The `Cox Process Example`_ is a more in-depth example of a model that can be built using this interface.

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   Pyro_GPyTorch_Low_Level.ipynb
   Cox_Process_Example.ipynb

.. _Pyro + GPyTorch High-Level Introduction:
  Pyro_GPyTorch_High_Level.ipynb

.. _Clustered Multitask GP Example:
  Clustered_Multitask_GP_Regression.ipynb

.. _Pyro + GPyTorch Low-Level Introduction:
  Pyro_GPyTorch_Low_Level.ipynb

.. _Cox Process Example:
  Cox_Process_Example.ipynb
