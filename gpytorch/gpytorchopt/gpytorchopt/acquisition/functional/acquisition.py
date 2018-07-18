#!/usr/bin/env python3

from gpytorch import Module
import torch
from torch.distributions import Normal


def expected_improvement(
    model: Module,
    candidate_set: torch.Tensor,
    f_best: float,
) -> torch.Tensor:
    """
    Parallel evaluation of single-point expected improvement, assumes
    maximization

    Args:
        model (Module): GP from gpytorch
        candidate_set (torch.Tensor): 'r x n' dimensional tensor where r is the number of
        points to evaluate, and n is the number of parameters
        fbest: previous best value
    """
    model.eval()
    model.likelihood.eval()

    pred = model.likelihood(model(candidate_set))

    mu = pred.mean().detach()
    sigma = pred.std().detach()

    u = (f_best - mu) / sigma
    m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
    ucdf = m.cdf(u)
    updf = torch.exp(m.log_prob(u))
    ei = sigma * (updf + u * ucdf)

    return ei


def posterior_mean(
    model: Module,
    candidate_set: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel evaluation of single-point posterior mean

    Args:
        model (Module): GP from gpytorch
        candidate_set (torch.Tensor): 'r x n' dimensional tensor where r is the number of
        points to evaluate, and n is the number of parameters
    """
    model.eval()
    model.likelihood.eval()
    pred_rv = model(candidate_set)
    mean = pred_rv.mean()
    return mean


def probability_of_improvement(
    model: Module,
    candidate_set: torch.Tensor,
    fbest: float,
) -> torch.Tensor:
    """
    Parallel evaluation of single-point probability of improvement, assumes
    maximization

    Args:
        model (Module): GP from gpytorch,
        candidate_set (torch.Tensor): 'r x n' dimensional tensor where r is the number of
        points to evaluate, and n is the number of parameters,
        fbest: previous best point
    """
    model.eval()
    model.likelihood.eval()
    normal = torch.distributions.Normal(0, 1)
    pred_rv = model(candidate_set)
    mean = pred_rv.mean()
    sd = pred_rv.covar().diag().sqrt()
    # TODO: ensure works with sd = 0.0
    u = (mean - fbest) / sd
    return normal.cdf(u)


def upper_confidence_bound(
    model: Module,
    candidate_set: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Parallel evaluation of single-point UCB, assumes maximization

    Args:
        model (Module): GP from gpytorch
        candidate_set (torch.Tensor): 'r x n' dimensional tensor where r is the number of
        points to evaluate, and n is the number of parameters
        beta: used to trade-off mean versus covariance
    """
    model.eval()
    model.likelihood.eval()
    pred_rv = model(candidate_set)
    mean = pred_rv.mean()
    var = pred_rv.covar().diag() * beta
    return mean + var.sqrt()


def max_value_entropy_search(
    model: Module,
    candidate_set: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    model.eval()
    model.likelihood.eval()

    pred = model.likelihood(model(candidate_set))

    mu = pred.mean().detach()
    sigma = pred.std().detach()

    # K samples of the posterior function f
    f_samples = pred.sample(num_samples)

    # K samples of y_star
    ys = f_samples.max(dim=0)[0]
    ysArray = ys.unsqueeze(0).expand(candidate_set.shape[0], num_samples)

    # compute gamma_y_star
    muArray = mu.unsqueeze(1).expand(candidate_set.shape[0], num_samples)
    sigmaArray = sigma.unsqueeze(1).expand(candidate_set.shape[0], num_samples)
    gamma = (ysArray - muArray) / sigmaArray

    # Compute the acquisition function of MES.
    m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))  # standard normal
    pdfgamma = torch.exp(m.log_prob(gamma))
    cdfgamma = m.cdf(gamma)

    mve = torch.mean(gamma * pdfgamma / (2 * cdfgamma) - torch.log(cdfgamma), dim=1)
    return mve
