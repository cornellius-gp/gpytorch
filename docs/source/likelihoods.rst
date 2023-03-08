.. role:: hidden
    :class: hidden-section

gpytorch.likelihoods
===================================

.. automodule:: gpytorch.likelihoods
.. currentmodule:: gpytorch.likelihoods


Likelihood
--------------------

.. autoclass:: Likelihood
   :special-members: __call__
   :members:


One-Dimensional Likelihoods
-----------------------------

Likelihoods for GPs that are distributions of scalar functions.
(I.e. for a specific :math:`\mathbf x` we expect that :math:`f(\mathbf x) \in \mathbb{R}`.)

One-dimensional likelihoods should extend :obj:`gpytoch.likelihoods._OneDimensionalLikelihood` to
reduce the variance when computing approximate GP objective functions.
(Variance reduction is accomplished by using 1D Gauss-Hermite quadrature rather than MC-integration).


:hidden:`GaussianLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianLikelihood
   :members:

:hidden:`GaussianLikelihoodWithMissingObs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianLikelihoodWithMissingObs
   :members:

:hidden:`FixedNoiseGaussianLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FixedNoiseGaussianLikelihood
   :members:


:hidden:`DirichletClassificationLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DirichletClassificationLikelihood
   :members:


:hidden:`BernoulliLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BernoulliLikelihood
   :members:


:hidden:`BetaLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BetaLikelihood
   :members:


:hidden:`LaplaceLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LaplaceLikelihood
   :members:


:hidden:`StudentTLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StudentTLikelihood
   :members:


Multi-Dimensional Likelihoods
-----------------------------

Likelihoods for GPs that are distributions of vector-valued functions.
(I.e. for a specific :math:`\mathbf x` we expect that :math:`f(\mathbf x) \in \mathbb{R}^t`,
where :math:`t` is the number of output dimensions.)


:hidden:`MultitaskGaussianLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultitaskGaussianLikelihood
   :members:


:hidden:`SoftmaxLikelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SoftmaxLikelihood
   :members:
