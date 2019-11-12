.. role:: hidden
    :class: hidden-section

gpytorch.distributions
===================================

GPyTorch distribution objects are essentially the same as torch distribution objects.
For the most part, GpyTorch relies on torch's distribution library.
However, we offer two custom distributions.

We implement a custom :obj:`~gpytorch.distributions.MultivariateNormal` that accepts
:obj:`~gpytorch.lazy.LazyTensor` objects for covariance matrices. This allows us to use custom
linear algebra operations, which makes this more efficient than PyTorch's MVN implementation.

In addition, we implement a :obj:`~gpytorch.distributions.MultitaskMultivariateNormal` which
can be used with multi-output Gaussian process models.

.. note::

  If Pyro is available, all GPyTorch distribution objects inherit Pyro's distribution methods
  as well.

.. automodule:: gpytorch.distributions
.. currentmodule:: gpytorch.distributions


Distribution
-----------------------------

.. autoclass:: Distribution
   :members:


MultivariateNormal
-----------------------------

.. autoclass:: MultivariateNormal
   :members:


MultitaskMultivariateNormal
----------------------------------

.. autoclass:: MultitaskMultivariateNormal
   :members:


Delta
----------------------------------

.. class:: Delta(v, log_density=0.0, event_dim=0, validate_args=None)

  (Borrowed from Pyro.) Degenerate discrete distribution (a single point).

  Discrete distribution that assigns probability one to the single element in
  its support. Delta distribution parameterized by a random choice should not
  be used with MCMC based inference, as doing so produces incorrect results.

  :param v: The single support element.
  :param log_density: An optional density for this Delta. This is useful to
    keep the class of Delta distributions closed under differentiable
    transformation.
  :param event_dim: Optional event dimension, defaults to zero.
  :type v: torch.Tensor
  :type log_density: torch.Tensor
  :type event_dim: int
