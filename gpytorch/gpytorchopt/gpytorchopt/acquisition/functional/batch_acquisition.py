#!/usr/bin/env python3

from gpytorch import Module
from math import pi, sqrt
import torch
import torch.nn.functional as F

from math import inf


def batch_simple_regret(
    x: torch.Tensor,
    model: Module,
    mc_samples: int=1000,
) -> torch.Tensor:
    # let's be paranoid
    x.requires_grad = True
    model.eval()
    val = model.forward(x).sample(mc_samples).max(0)[0].mean()
    return val


def batch_probability_of_improvement(
    x: torch.Tensor,
    model: Module,
    alpha: torch.Tensor,
    mc_samples: int=1000
) -> torch.Tensor:
    # let's be paranoid
    x.requires_grad = True
    model.eval()
    val = F.sigmoid(model.forward(x).sample(mc_samples).max(0)[0] - alpha).mean()
    return val


def batch_expected_improvement(
    x: torch.Tensor,
    model: Module,
    alpha: torch.Tensor,
    mc_samples: int=1000,
) -> torch.Tensor:
    # let's be paranoid
    x.requires_grad = True
    model.eval()
    val = (model.forward(x).sample(mc_samples).max(0)[0] - alpha).clamp(0, inf).mean()
    return val


def batch_upper_confidence_bound(
    x: torch.Tensor,
    model: Module,
    beta: float,
    mc_samples: int=1000
) -> torch.Tensor:
    x.requires_grad = True
    model.eval()
    mvn = model.forward(x)
    val = (
        sqrt(beta * pi / 2) * mvn.covar().zero_mean_mvn_samples(mc_samples).abs() +
        mvn._mean.view(-1, 1)
    ).max(0)[0].mean()
    return val
