from math import pi
from typing import Optional

import torch

from ..distributions import MultitaskMultivariateNormal, MultivariateNormal

pi = torch.tensor(pi)


def mean_absolute_error(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):
    """
    Mean absolute error.
    """
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    return torch.abs(pred_dist.mean - test_y).mean(dim=combine_dim)


def mean_squared_error(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
    squared: bool = True,
):
    """
    Mean squared error.
    """
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    res = torch.square(pred_dist.mean - test_y).mean(dim=combine_dim)
    if not squared:
        return res**0.5
    return res


def standardized_mean_squared_error(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):
    """Standardized mean squared error.

    Standardizes the mean squared error by the variance of the test data.
    """
    return mean_squared_error(pred_dist, test_y, squared=True) / test_y.var()


def negative_log_predictive_density(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):
    """Negative log predictive density.

    Computes the negative predictive log density normalized by the size of the test data.
    """
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    return -pred_dist.log_prob(test_y) / test_y.shape[combine_dim]


def mean_standardized_log_loss(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
    train_y: Optional[torch.Tensor] = None,
):
    """
    Mean standardized log loss.

    Computes the average *standardized* log loss, which subtracts the loss obtained
    under the trivial model, which predicts with the mean and variance of the training
    data, from the mean log loss. See p.23 of Rasmussen and Williams (2006).

    If no training data is supplied, the mean log loss is computed.
    """
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1

    f_mean = pred_dist.mean
    f_var = pred_dist.variance
    loss_model = (0.5 * torch.log(2 * pi * f_var) + torch.square(test_y - f_mean) / (2 * f_var)).mean(dim=combine_dim)
    res = loss_model

    if train_y is not None:
        data_mean = train_y.mean(dim=combine_dim)
        data_var = train_y.var()
        loss_trivial_model = (
            0.5 * torch.log(2 * pi * data_var) + torch.square(test_y - data_mean) / (2 * data_var)
        ).mean(dim=combine_dim)
        res = res - loss_trivial_model

    return res


def quantile_coverage_error(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
    quantile: float = 95.0,
):
    """
    Quantile coverage error.
    """
    if quantile <= 0 or quantile >= 100:
        raise NotImplementedError("Quantile must be between 0 and 100")
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * (quantile / 100)))
    lower = pred_dist.mean - deviation * pred_dist.stddev
    upper = pred_dist.mean + deviation * pred_dist.stddev
    n_samples_within_bounds = ((test_y > lower) * (test_y < upper)).sum(combine_dim)
    fraction = n_samples_within_bounds / test_y.shape[combine_dim]
    return torch.abs(fraction - quantile / 100)


def kl_divergence(q: MultivariateNormal, p: MultivariateNormal):
    """Kullback-Leibler Divergence.

    See for example: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

    :param q: First distributional argument. Can be a stack of multivariate normals.
    :param p: Second distributional argument. Can be a stack of multivariate normals.
    """

    # Error checking
    if not q.event_shape == p.event_shape:
        raise ValueError("Multivariate normal distributions must have same event shape.")

    # Compute Cholesky decompositions
    q_cov_chol = torch.linalg.cholesky(q.covariance_matrix, upper=False)
    p_cov_chol = torch.linalg.cholesky(p.covariance_matrix, upper=False)

    # Forward substitution
    M = torch.linalg.solve_triangular(p_cov_chol, q_cov_chol, upper=False, left=True)
    mean_diff = p.mean - q.mean
    y = torch.squeeze(
        torch.linalg.solve_triangular(p_cov_chol, torch.unsqueeze(mean_diff, dim=-1), upper=False, left=True),
        dim=-1,
    )

    # Individual terms
    trace_term = torch.sum(M**2, dim=(-2, -1))
    mahanalobis_distance_term = torch.sum(y**2, dim=-1)
    logdet_term = torch.sum(
        torch.log(torch.diagonal(p_cov_chol, dim1=-2, dim2=-1))
        - torch.log(torch.diagonal(q_cov_chol, dim1=-2, dim2=-1)),
        dim=-1,
    )

    return 0.5 * (trace_term - q.event_shape[0] + mahanalobis_distance_term) + logdet_term


def _symsqrt(matrix):
    """Compute the square root of a positive definite matrix."""

    # SVD
    _, s, v = matrix.svd()

    # Truncate small components
    above_cutoff = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = above_cutoff.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            above_cutoff = above_cutoff[..., :common]
    if unbalanced:
        s = s.where(above_cutoff, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def wasserstein(q: MultivariateNormal, p: MultivariateNormal, order=2):
    """Wasserstein distance or earth mover's distance.

    See for example: https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/

    :param q: First distributional argument. Can be a stack of multivariate normals.
    :param p: Second distributional argument. Can be a stack of multivariate normals.
    :param order: Order of the Wasserstein distance.
    """

    # Error checking
    if not q.event_shape == p.event_shape:
        raise ValueError("Multivariate normal distributions must have same event shape.")
    if order != 2:
        raise NotImplementedError("Only order p=2 is supported currently.")

    # Compute individual terms
    q_cov_sqrt = _symsqrt(q.covariance_matrix)
    if p.batch_shape == torch.Size([]):
        mixed_cov_matrix = q_cov_sqrt @ p.covariance_matrix @ q_cov_sqrt
        mixed_cov_sqrt = _symsqrt(mixed_cov_matrix)
        mixed_trace_term = torch.trace(mixed_cov_sqrt)
    else:
        mixed_cov_matrix = torch.bmm(q_cov_sqrt, torch.bmm(p.covariance_matrix, q_cov_sqrt))
        mixed_cov_sqrt = _symsqrt(mixed_cov_matrix)
        mixed_trace_term = torch.vmap(torch.trace)(mixed_cov_sqrt)

    mean_sq_distance_term = torch.sum((q.mean - p.mean) ** 2, dim=(-1))

    return (
        mean_sq_distance_term + torch.sum(q.variance, dim=-1) + torch.sum(p.variance, dim=-1) - 2.0 * mixed_trace_term
    )
