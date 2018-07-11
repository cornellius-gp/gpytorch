#!/usr/bin/env python3

import torch
from gpytorch import Module


def posterior_mean(
    model: Module, X: torch.Tensor
) -> torch.Tensor:
    """
        Parallel evaluation of single-point posterior mean

        Args:
            model (Module): GP from gpytorch
            X (torch.Tensor): 'r x n' dimensional tensor where r is the number of
            points to evaluate, and n is the number of parameters
    """
    pred_rv = model(X)
    mean = pred_rv.mean()
    return mean
