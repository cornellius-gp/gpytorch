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
    under the trivial model which predicts with the mean and variance of the training
    data from the mean log loss. See p.23 of Rasmussen and Williams (2006).

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
