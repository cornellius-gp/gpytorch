#!/usr/bin/env python3

import torch

from ..distributions import MultivariateNormal
from ..utils.memoize import add_to_cache, cached
from ._variational_strategy import _VariationalStrategy
from .delta_variational_distribution import DeltaVariationalDistribution


class OrthogonallyDecoupledVariationalStrategy(_VariationalStrategy):
    r"""
    Implements orthogonally decoupled VGPs as defined in `Salimbeni et al. (2018)`_.
    This variational strategy uses a different set of inducing points for the mean and covariance functions.
    The idea is to use more inducing points for the (computationally efficient) mean and fewer inducing points for the
    (computationally expensive) covaraince.

    This variational strategy defines the inducing points/:obj:`~gpytorch.variational._VariationalDistribution`
    for the mean function.
    It then wraps a different :obj:`~gpytorch.variational._VariationalStrategy` which
    defines the covariance inducing points.

    :param ~gpytorch.variational._VariationalStrategy covar_variational_strategy:
        The variational strategy for the covariance term.
    :param torch.Tensor inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`

    Example:
        >>> mean_inducing_points = torch.randn(1000, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
        >>> covar_inducing_points = torch.randn(100, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
        >>>
        >>> covar_variational_strategy = gpytorch.variational.VariationalStrategy(
        >>>     model, covar_inducing_points,
        >>>     gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2)),
        >>>     learn_inducing_locations=True
        >>> )
        >>>
        >>> variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        >>>     covar_variational_strategy, mean_inducing_points,
        >>>     gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2)),
        >>> )

    .. _Salimbeni et al. (2018):
        https://arxiv.org/abs/1809.08820
    """

    def __init__(self, covar_variational_strategy, inducing_points, variational_distribution):
        if not isinstance(variational_distribution, DeltaVariationalDistribution):
            raise NotImplementedError(
                "OrthogonallyDecoupledVariationalStrategy currently works with DeltaVariationalDistribution"
            )

        super().__init__(
            covar_variational_strategy, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        self.base_variational_strategy = covar_variational_strategy

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        out = self.model(self.inducing_points)
        res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter())
        return res

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        if variational_inducing_covar is not None:
            raise NotImplementedError(
                "OrthogonallyDecoupledVariationalStrategy currently works with DeltaVariationalDistribution"
            )

        num_data = x.size(-2)
        full_output = self.model(torch.cat([x, inducing_points], dim=-2), **kwargs)
        full_mean = full_output.mean
        full_covar = full_output.lazy_covariance_matrix

        if self.training:
            induc_mean = full_mean[..., num_data:]
            induc_induc_covar = full_covar[..., num_data:, num_data:]
            prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)
            add_to_cache(self, "prior_distribution_memo", prior_dist)

        test_mean = full_mean[..., :num_data]
        data_induc_covar = full_covar[..., :num_data, num_data:]
        predictive_mean = (data_induc_covar @ inducing_values.unsqueeze(-1)).squeeze(-1).add(test_mean)
        predictive_covar = full_covar[..., :num_data, :num_data]

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def kl_divergence(self):
        mean = self.variational_distribution.mean
        induc_induc_covar = self.prior_distribution.lazy_covariance_matrix
        kl = self.model.kl_divergence() + ((induc_induc_covar @ mean.unsqueeze(-1)).squeeze(-1) * mean).sum(-1).mul(0.5)
        return kl
