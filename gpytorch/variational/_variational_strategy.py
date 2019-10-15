#!/usr/bin/env python3

import torch
from ..module import Module
from ..utils.memoize import cached
from abc import ABC, abstractproperty, abstractmethod


class _VariationalStrategy(Module, ABC):
    """
    _VariationalStrategy objects control how certain aspects of variational inference should be performed.
    In particular, they define two methods that get used during variational inference:

    # The :func:`~gpytorch.variational._VariationalStrategy.prior_distribution` method determines how to compute the
      GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
      this is done simply by calling the user defined GP prior on the inducing point data directly.
    # The :func:`~gpytorch.variational._VariationalStrategy.forward` method determines how to marginalize out the
      inducing point function values. Specifically, forward defines how to transform a variational distribution
      over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
      specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

    In GPyTorch, we currently support two categories of this latter functionality. In scenarios where the
    inducing points are learned (or set to be exactly the training data), we apply the derivation in
    Hensman et al., 2015 to exactly marginalize out the variational distribution. When the inducing points
    are constrained to a grid, we apply the derivation in Wilson et al., 2016 and exploit a
    deterministic relationship between f and u.

    Args:
        :attr:`model` (:obj:`gpytorch.model.VariationalGP`):
            Model this strategy is applied to. Typically passed in when the VariationalStrategy is created
            in the __init__ method of the user defined model.
        :attr:`inducing_points` (:obj:`torch.Tensor`):
            Tensor containing a set of inducing points to use for variational inference.
        :attr:`variational_distribution` (:obj:`gpytorch.variational.VariationalDistribution`):
            A VariationalDistribution object that represents the form of the variational distribution :math:`q(u)`
        :attr:`learn_inducing_locations` (bool - default True):
            Whether or not the inducing point locations should be learned (e.g. SVGP).
    """
    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=True):
        super().__init__()

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        if learn_inducing_locations:
            self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("inducing_points", inducing_points)

        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    @abstractproperty
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.

        Returns:
            :obj:`gpytorch.distributions.MultivariateNormal`: The distribution p(u)
        """
        raise NotImplementedError

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        return self._variational_distribution()

    @abstractmethod
    def forward(self, x):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        Args:
            x (torch.tensor): Locations x to get the variational posterior of the function values at.
        Returns:
            :obj:`gpytorch.distributions.MultivariateNormal`: The distribution q(f|x)
        """
        raise NotImplementedError

    def train(self, mode=True):
        # Make sure we are clearing the cache if we change modes
        if (self.training and not mode) or mode:
            if hasattr(self, "_memoize_cache"):
                delattr(self, "_memoize_cache")
        return super().train(mode=mode)

    def __call__(self, x):
        # Delete previously cached items from the training distribution
        if self.training:
            if hasattr(self, "_memoize_cache"):
                delattr(self, "_memoize_cache")
                self._memoize_cache = dict()
        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            self._variational_distribution.initialize_variational_distribution(prior_dist)
            self.variational_params_initialized.fill_(1)
        return super().__call__(x)
