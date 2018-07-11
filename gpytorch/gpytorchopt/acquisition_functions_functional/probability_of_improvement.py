#!/usr/bin/env python3

import torch
from gpytorch import Module


def probability_of_improvement(
    model: Module, X: torch.Tensor, fbest: float
) -> torch.Tensor:
    """
        Parallel evaluation of single-point probability of improvement, assumes
        maximization

        Args:
            model (Module): GP from gpytorch,
            X (torch.Tensor): 'r x n' dimensional tensor where r is the number of
            points to evaluate, and n is the number of parameters,
            fbest: previous best point
    """
    normal = torch.distributions.Normal(0, 1)
    pred_rv = model(X)
    mean = pred_rv.mean()
    sd = pred_rv.covar().diag().sqrt()
    # TODO: ensure works with sd = 0.0
    u = (mean - fbest) / sd
    return normal.cdf(u)
