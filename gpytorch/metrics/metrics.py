from math import pi

import torch

from ..distributions import MultivariateNormal

pi = torch.tensor(pi)


def mean_absolute_error(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):
    """
    Mean Absolute Error.

    """
    return torch.abs(pred_dist.mean - test_y).mean(dim=0)


def mean_squared_error(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
    squared: bool = True,
):
    """
    Mean Squared Error.

    """
    res = torch.square(pred_dist.mean - test_y).mean(dim=0)
    if not squared:
        return res ** 0.5
    return res


def negative_log_predictive_density(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):

    return -pred_dist.log_prob(test_y) / test_y.shape[0]


def mean_standardized_log_loss(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):
    """
    Mean Standardized Log Loss.

    Reference: Page No. 23,
    Gaussian Processes for Machine Learning,
    Carl Edward Rasmussen and Christopher K. I. Williams,
    The MIT Press, 2006. ISBN 0-262-18253-X
    """

    f_mean = pred_dist.mean
    f_var = pred_dist.variance
    return 0.5 * (torch.log(2 * pi * f_var) + torch.square(test_y - f_mean) / (2 * f_var)).mean(dim=0)


def quantile_coverage_error(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
    quantile: float = 95,
):
    """
    Quantile coverage error.

    """

    assert 0 <= quantile <= 100, "Quantile must be between 0 and 100"

    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * (quantile / 100)))
    lower = pred_dist.mean - deviation * pred_dist.stddev
    upper = pred_dist.mean + deviation * pred_dist.stddev
    n_samples_within_bounds = ((test_y > lower) * (test_y < upper)).sum(0)
    fraction = n_samples_within_bounds / test_y.shape[0]
    return torch.abs(fraction - quantile / 100)


def average_coverage_error(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
    n_bins: int = 20,
):
    """
    Average Coverage error.

    """
    bins = torch.linspace(100 / n_bins, 100 - 100 / n_bins, n_bins - 1)
    res = 0
    for quantile in bins:
        res += quantile_coverage_error(pred_dist, test_y, quantile)
    return res / n_bins
