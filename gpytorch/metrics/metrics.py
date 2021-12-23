from math import pi

import torch

from ..distributions import MultivariateNormal

pi = torch.tensor(pi)


def negative_log_predictive_density(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
) -> float:

    with torch.no_grad():
        return -pred_dist.log_prob(test_y).item()


def mean_standardized_log_loss(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
) -> float:
    """
    Reference: GPML book
    """

    with torch.no_grad():
        f_mean = pred_dist.mean
        f_var = pred_dist.variance
        return 0.5 * (torch.log(2 * pi * f_var) + torch.square(test_y - f_mean) / (2 * f_var)).mean().item()


def coverage_error(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
) -> float:
    """
    Coverage error for 95% confidence intervals.
    TODO: Find a good reference to improvise this metric.
    """

    with torch.no_grad():
        lower, upper = pred_dist.confidence_region()
        n_samples_within_bounds = ((test_y > lower) * (test_y < upper)).sum()
        fraction = (n_samples_within_bounds / test_y.shape[0]).item()
        return abs(0.9545 - fraction)
