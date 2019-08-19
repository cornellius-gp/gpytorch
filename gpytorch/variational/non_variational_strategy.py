#!/usr/bin/env python3

import torch
from .. import settings
from ..lazy import DiagLazyTensor, RootLazyTensor
from ..module import Module
from ..distributions import MultivariateNormal
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.cholesky import psd_safe_cholesky
from ..utils.memoize import cached
from ..utils.lanczos import lanczos_tridiag


class NonVariationalStrategy(Module):
    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=False):
        super(NonVariationalStrategy, self).__init__()
        object.__setattr__(self, "model", model)

        inducing_points = inducing_points.clone()

        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        if learn_inducing_locations:
            self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("inducing_points", inducing_points)

        self.variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.
        """
        res = MultivariateNormal(
            torch.zeros_like(inducing_points[..., 0]),
            DiagLazyTensor(torch.ones_like(inducing_points[..., 0]))
        )
        return res

    def kl_divergence(self):
        return torch.tensor(0., dtype=self.inducing_points.dtype, device=self.inducing_points.device)

    def initialize_variational_dist(self):
        self.variational_distribution.variational_distribution.mean.data.fill_(0)

    def forward(self, input):
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
        variational_mean = self.variational_distribution.variational_distribution.mean

        num_induc = self.inducing_points.size(-2)
        full_inputs = torch.cat([self.inducing_points, input], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        test_mean = full_output.mean[..., num_induc:]
        L = full_covar[..., :num_induc, :num_induc].add_jitter().cholesky().evaluate()
        Linv = torch.triangular_solve(torch.eye(L.size(-1), dtype=L.dtype, device=L.device), L, upper=False)[0]
        cross_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        scaled_cross_covar = Linv @ cross_covar
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        function_dist = MultivariateNormal(
            torch.squeeze(scaled_cross_covar.transpose(-1, -2) @ variational_mean.unsqueeze(-1)),
            data_data_covar + RootLazyTensor(scaled_cross_covar.transpose(-1, -2)).mul(-1)
        )
        return function_dist

    def __call__(self, x):
        if not self.variational_params_initialized.item():
            self.initialize_variational_dist()
            self.variational_params_initialized.fill_(1)
        if self.training:
            if hasattr(self, "_memoize_cache"):
                delattr(self, "_memoize_cache")
                self._memoize_cache = dict()

        return super(NonVariationalStrategy, self).__call__(x)
