from math import pi

import torch

from ..distributions import MultivariateNormal

pi = torch.tensor(pi)


def negative_log_predictive_density(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
):

    with torch.no_grad():
        return -pred_dist.log_prob(test_y).item()


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

    with torch.no_grad():
        f_mean = pred_dist.mean
        f_var = pred_dist.variance
        return 0.5 * (torch.log(2 * pi * f_var) + torch.square(test_y - f_mean) / (2 * f_var)).mean().item()


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
    with torch.no_grad():
        deviation = standard_normal.icdf(torch.tensor(0.5 + 0.5 * (quantile / 100)))
        lower = pred_dist.mean - deviation * pred_dist.stddev
        upper = pred_dist.mean + deviation * pred_dist.stddev
        n_samples_within_bounds = ((test_y > lower) * (test_y < upper)).sum(0)
        fraction = n_samples_within_bounds / test_y.shape[0]
        return abs(fraction - quantile / 100)


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