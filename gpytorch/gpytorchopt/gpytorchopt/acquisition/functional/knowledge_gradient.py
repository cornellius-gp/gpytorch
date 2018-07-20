#!/usr/bin/env python3

import numpy as np
import torch
from gpytorch import Module


def knowledge_gradient(model: Module, noise_model: Module, X: torch.Tensor, Xprev: torch.Tensor) -> torch.Tensor:
    """
        Parallel evaluation of single-point KG, assumes maximization

        Args:
            model (Module): GP from gpytorch
            X (torch.Tensor): 'r x n' dimensional tensor where r is the number
            of points to evaluate, and n is the number of parameters
            Xprev (torch.Tensor): 'm x n' dimensional tensor where these m
            points are considered as possible best points and n is the number
            of parameters.  A single point in X is compared against all points
            in Xprev.
    """
    # Get predictive mean and covariance across test and train
    Xall = torch.cat((X, Xprev), dim=1)
    pred_rv = model(Xall)

    # All means and covariances with X
    mean = pred_rv.mean()  # (r + m)
    all_covar = pred_rv.covar()  # lazy variable: (r + m) x (r + m)
    covar_with_point = torch.stack([all_covar[i, X.shape[0] :] for i in range(X.shape[0])])  # r x m
    var_point = all_covar.diag()[: X.shape[0]].unsqueeze(1)  # r x 1

    covar_with_point = torch.cat([var_point, covar_with_point], dim=1)  # r x (1 + m)

    # Deterministic prediction of noise at new points.  Sample size adjustment
    # is assumed to happen implicitly through a feature in X.
    mean_noise = noise_model(X).mean().squeeze(1)  # r
    denom = (torch.max(mean_noise, torch.zeros_like(mean_noise)) + covar_with_point[:, 0]).unsqueeze(1).sqrt()  # r x 1

    tmp = mean[X.shape[0] :]  # m
    a = torch.stack([torch.cat([mean[i : i + 1], tmp]) for i in range(X.shape[0])])  # (1 + m)  # r x (1 + m)
    b = covar_with_point / denom  # r x (1 + m)

    # TODO: figure out how to implement below
    a_filter, b_filter = _filter_a_and_b(a, b)
    c, A = _compute_c_and_A(a_filter, b_filter)

    b_filter = b_filter[A]
    c = -torch.abs(c)

    normal = torch.distributions.Normal(0, 1)
    # TODO: ensure works with c = infinity
    f_c = c * normal.cdf(c) + normal.log_prob(c).exp()

    result = torch.zeros_like(mean_noise)  # r
    # TODO: figure out how to write this in vectorized form
    for j in range(b_filter.shape[1] - 1):
        result += (b_filter[:, j + 1] - b_filter[:, j]) * f_c[:, j]

    # Unlike KGCP we only consider an arm as viable once it has been submitted
    a_x = a[:, 0]
    a_prev = torch.max(a[:, 1:], dim=1)
    result += torch.max(a_x - a_prev, torch.zeros(1))

    return result


def _filter_a_and_b(a, b):
    assert False, "Needs to be converted"
    b_a_pairs = list(zip(b, a))
    b_a_pairs.sort()

    filtered_b_a_pairs = []
    for j in range(len(b_a_pairs) - 1):
        if b_a_pairs[j][0] != b_a_pairs[j + 1][0]:
            filtered_b_a_pairs.append(b_a_pairs[j])
    filtered_b_a_pairs.append(b_a_pairs[-1])

    return torch.Tensor([tmp[0] for tmp in filtered_b_a_pairs], [tmp[1] for tmp in filtered_b_a_pairs])


def _compute_c_and_A(a, b):
    assert False, "Needs to be converted"
    c = np.zeros(len(b) + 1)
    c[0] = -np.inf
    c[1] = np.inf
    A = [1]
    for i in range(1, len(b)):
        c[i + 1] = np.inf
        loopdone = False
        while not loopdone:
            j = A[-1]
            # Algorithm is one-indexed so subtracting one from a and b
            # indexes here.  Note this is important or otherwise a[i] will
            # be out-of-bounds for the last element of range(1, M)
            c[j] = (a[j - 1] - a[i]) / (b[i] - b[j - 1])
            if len(A) != 1 and c[j] <= c[A[-2]]:
                A.pop()
            else:
                loopdone = True
        A.append(i + 1)
    # Output is currently in the form of
    # i in A and z in (c[i-1], c[i]) => i - 1 is the best arm
    # where the -1 here is due to the zero-indexing of a and b

    # We define A_tilde the zero-indexed form such that
    # i in A_tilde and z in (c[i], c[i + 1]) => i is the best arm
    A_tilde = torch.Tensor([x - 1 for x in A])

    # Now we define c_tilde as a zero-indexed form of c such that
    # i in A_tilde and z in (c_tilde[i-1], c_tilde[i]) => i is the best arm
    # where we just need c_tilde[i] for i in A_tilde in what follows.
    c_tilde = torch.Tensor([c[i + 1] for i in A_tilde])

    return c_tilde, A_tilde
