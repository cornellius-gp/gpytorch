#!/usr/bin/env python3

from abc import ABC, abstractproperty

import torch

from .. import settings
from ..distributions import Delta, MultivariateNormal
from ..module import Module
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.memoize import cached, clear_cache_hook, add_to_cache


class _AmortizedVariationalStrategy(Module, ABC):
    """
    Abstract base class for all Variational Strategies.
    """

    def __init__(self, model, inducing_point_model, variational_distribution_model):
        super().__init__()

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points
        self.inducing_point_model = inducing_point_model

        # Variational "distribution"
        self._variational_distribution_model = variational_distribution_model

    def _clear_cache(self):
        clear_cache_hook(self)

    def _expand_inputs(self, x, inducing_points):
        """
        Pre-processing step in __call__ to make x the same batch_shape as the inducing points
        """
        batch_shape = _mul_broadcast_shape(inducing_points.shape[:-2], x.shape[:-2])
        inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
        x = x.expand(*batch_shape, *x.shape[-2:])
        return x, inducing_points

    @abstractproperty
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`p( \mathbf u)`
        """
        raise NotImplementedError

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        raise RuntimeError("Well we should never get here.")

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        :param torch.Tensor x: Locations :math:`\mathbf X` to get the
            variational posterior of the function values at.
        :param torch.Tensor inducing_points: Locations :math:`\mathbf Z` of the inducing points
        :param torch.Tensor inducing_values: Samples of the inducing function values :math:`\mathbf u`
            (or the mean of the distribution :math:`q(\mathbf u)` if q is a Gaussian.
        :param ~gpytorch.lazy.LazyTensor variational_inducing_covar: If the distribuiton :math:`q(\mathbf u)`
            is Gaussian, then this variable is the covariance matrix of that Gaussian. Otherwise, it will be
            :attr:`None`.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`q( \mathbf f(\mathbf X))`
        """
        raise NotImplementedError

    def kl_divergence(self):
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.

        :rtype: torch.Tensor
        """
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence

    def __call__(self, x, prior=False):
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x)

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()

        # Ensure inducing_points and x are the same size

        # HACK: transformer layers require >= 3 dimensional inputs
        if x.ndim < 3:
            inducing_points = self.inducing_point_model(x.unsqueeze(-3)).squeeze(-3)
        else:
            inducing_points = self.inducing_point_model(x)

        # Reshape for batch dimensions
        if inducing_points.shape[:-2] != x.shape[:-2]:
            x, inducing_points = self._expand_inputs(x, inducing_points)

        # Get p(u)/q(u)
        variational_dist_u = self._variational_distribution_model(inducing_points)

        # HACK: Not really the right way to do this, but...
        add_to_cache(self, "variational_distribution_memo", variational_dist_u)

        # Get q(f)
        if isinstance(variational_dist_u, MultivariateNormal):
            return super().__call__(
                x,
                inducing_points,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
            )
        elif isinstance(variational_dist_u, Delta):
            return super().__call__(
                x, inducing_points, inducing_values=variational_dist_u.mean, variational_inducing_covar=None
            )
        else:
            raise RuntimeError(
                f"Invalid variational distribuition ({type(variational_dist_u)}). "
                "Expected a multivariate normal or a delta distribution."
            )
