#!/usr/bin/env python3

import torch
from gpytorch import Module


def upper_confidence_bound(
    model: Module, X: torch.Tensor, beta: float
) -> torch.Tensor:
    """
        Parallel evaluation of single-point UCB, assumes maximization

        Args:
            model (Module): GP from gpytorch
            X (torch.Tensor): 'r x n' dimensional tensor where r is the number of
            points to evaluate, and n is the number of parameters
            beta: used to trade-off mean versus covariance
    """
    pred_rv = model(X)
    mean = pred_rv.mean()
    var = pred_rv.covar().diag() * beta
    return mean + var.sqrt()
