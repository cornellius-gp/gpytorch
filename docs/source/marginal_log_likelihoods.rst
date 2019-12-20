.. role:: hidden
    :class: hidden-section

gpytorch.mlls
===================================

These are modules to compute (or approximate/bound) the marginal log likelihood
(MLL) of the GP model when applied to data.  I.e., given a GP :math:`f \sim
\mathcal{GP}(\mu, K)`, and data :math:`\mathbf X, \mathbf y`, these modules
compute/approximate

.. math::

   \begin{equation*}
      \mathcal{L} = p_f(\mathbf y \! \mid \! \mathbf X)
      = \int p \left( \mathbf y \! \mid \! f(\mathbf X) \right) \: p(f(\mathbf X) \! \mid \! \mathbf X) \: d f
   \end{equation*}

This is computed exactly when the GP inference is computed exactly (e.g. regression w/ a Gaussian likelihood).
It is approximated/bounded for GP models that use approximate inference.

These models are typically used as the "loss" functions for GP models (though note that the output of
these functions must be negated for optimization).

.. automodule:: gpytorch.mlls
.. currentmodule:: gpytorch.mlls


Exact GP Inference
-----------------------------

These are MLLs for use with :obj:`~gpytorch.models.ExactGP` modules. They compute the MLL exactly.

:hidden:`ExactMarginalLogLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExactMarginalLogLikelihood
   :members:


Approximate GP Inference
-----------------------------------

These are MLLs for use with :obj:`~gpytorch.models.ApproximateGP` modules. They are designed for
when exact inference is intractable (either when the likelihood is non-Gaussian likelihood, or when
there is too much data for an ExactGP model).

:hidden:`VariationalELBO`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VariationalELBO
   :members:

:hidden:`PredictiveLogLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PredictiveLogLikelihood
   :members:

:hidden:`GammaRobustVariationalELBO`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GammaRobustVariationalELBO
   :members:

:hidden:`DeepApproximateMLL`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DeepApproximateMLL
   :members:
