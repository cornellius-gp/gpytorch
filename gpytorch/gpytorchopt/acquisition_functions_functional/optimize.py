#! /usr/bin/env python3

from typing import Callable

import torch
from torch.optim import LBFGS


def random_restarts(
    gp: torch.nn.Module,
    function_to_optimize: Callable,
    n_cols: int,
    n_restarts: int = 3,
    max_iter: int = 100,
    optimizer: torch.optim.Optimizer = LBFGS,
    learning_rate: float = 0.01,
):
    # randomly sample restart points
    X = torch.rand(n_restarts, n_cols)
    X.requires_grad = True
    optimizer = optimizer(params=[X], lr=learning_rate)
    trajectory = []
    fvals = []
    for _ in range(max_iter):

        def closure():
            optimizer.zero_grad()
            val = function_to_optimize(gp, X)
            # maximizing sum is fine since each component is independent
            val = val.sum()
            fvals.append(val.item())
            (-val).backward(retain_graph=True)
            return -val

        trajectory.append(X.detach().numpy())
        optimizer.step(closure)
    trajectory.append(X.detach().numpy())
    trajectory = torch.tensor(trajectory)
    fvals = torch.tensor(fvals)

    return trajectory, fvals
