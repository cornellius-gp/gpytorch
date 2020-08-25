.. role:: hidden
    :class: hidden-section

gpytorch.variational
===================================

.. automodule:: gpytorch.variational
.. currentmodule:: gpytorch.variational

There are many possible variants of variational/approximate GPs.
GPyTorch makes use of 3 composible objects that make it possible to implement
most GP approximations:

- :obj:`VariationalDistribution`, which define the form of the approximate inducing value
  posterior :math:`q(\mathbf u)`.
- :obj:`VarationalStrategies`, which define how to compute :math:`q(\mathbf f(\mathbf X))` from
  :math:`q(\mathbf u)`.
- :obj:`~gpytorch.mlls._ApproximateMarginalLogLikelihood`, which defines the objective function
  to learn the approximate posterior (e.g. variational ELBO).

All three of these objects should be used in conjunction with a :obj:`gpytorch.models.ApproximateGP`
model.


Variational Strategies
-----------------------------

VariationalStrategy objects control how certain aspects of variational inference should be performed.
In particular, they define two methods that get used during variational inference:

- The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
  GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
  this is done simply by calling the user defined GP prior on the inducing point data directly.
- The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
  inducing point function values. Specifically, forward defines how to transform a variational distribution
  over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
  specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

In GPyTorch, we currently support two categories of this latter functionality. In scenarios where the
inducing points are learned (or set to be exactly the training data), we apply the derivation in
Hensman et al., 2015 to exactly marginalize out the variational distribution. When the inducing points
are constrained to a grid, we apply the derivation in Wilson et al., 2016 and exploit a
deterministic relationship between :math:`\mathbf f` and :math:`\mathbf u`.

:hidden:`_VariationalStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: _VariationalStrategy
   :members:


:hidden:`VariationalStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VariationalStrategy
   :members:


:hidden:`BatchDecoupledVariationalStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BatchDecoupledVariationalStrategy
   :members:


:hidden:`OrthogonallyDecoupledVariationalStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: OrthogonallyDecoupledVariationalStrategy
   :members:


:hidden:`UnwhitenedVariationalStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UnwhitenedVariationalStrategy
   :members:


:hidden:`GridInterpolationVariationalStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GridInterpolationVariationalStrategy
   :members:


Variational Strategies for Multi-Output Functions
---------------------------------------------------

These are special :obj:`~gpytorch.variational._VariationalStrategy` objects that return
:obj:`~gpytorch.distributions.MultitaskMultivariateNormal` distributions. Each of these objects
acts on a batch of approximate GPs.


:hidden:`IndependentMultitaskVariationalStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: IndependentMultitaskVariationalStrategy
   :members:

:hidden:`LMCVariationalStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LMCVariationalStrategy
   :members:


Variational Distributions
-----------------------------

VariationalDistribution objects represent the variational distribution
:math:`q(\mathbf u)` over a set of inducing points for GPs.  Typically the
distributions are some sort of parameterization of a multivariate normal
distributions.

:hidden:`_VariationalDistribution`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: _VariationalDistribution
   :members:


:hidden:`CholeskyVariationalDistribution`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CholeskyVariationalDistribution
   :members:


:hidden:`DeltaVariationalDistribution`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DeltaVariationalDistribution
   :members:


:hidden:`MeanFieldVariationalDistribution`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MeanFieldVariationalDistribution
   :members:


Variational Distributions for Natural Gradient Descent
---------------------------------------------------------

Special :obj:`~_VariationalDistribution` objects designed specifically for
use with natural gradient descent techniques.

If the variational distribution is defined by :math:`\mathcal{N}(\mathbf m, \mathbf S)`, then
a :obj:`~gpytorch.variational.NaturalVariationalDistribution` uses the parameterization:

.. math::

    \begin{align*}
        \boldsymbol \theta_\text{vec} &= \mathbf S^{-1} \mathbf m
        \\
        \boldsymbol \Theta_\text{mat} &= -\frac{1}{2} \mathbf S^{-1}.
    \end{align*}

The gradients with respect to the variational parameters calculated by this
class are instead the **natural** gradients. Thus, optimising its
parameters using gradient descent (:obj:`~torch.optim.SGDOptimizer`)
becomes natural gradient descent (see e.g. `Salimbeni et al., 2018`_).

GPyTorch offers several :obj:`_NaturalVariationalDistribution` classes, each of which uses a
different representation of the natural parameters. The different parameterizations trade off
speed and stability.

.. note::
    Natural gradient descent is very stable with variational regression,
    and fast: if the hyperparameters are fixed, the variational parameters
    converge in 1 iteration. However, it can be unstable with non-conjugate
    likelihoods and alternative objective functions.

.. _Salimbeni et al., 2018:
    https://arxiv.org/abs/1803.09151


:hidden:`NaturalVariationalDistribution`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NaturalVariationalDistribution
   :members:


:hidden:`TrilNaturalVariationalDistribution`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TrilNaturalVariationalDistribution
   :members:
