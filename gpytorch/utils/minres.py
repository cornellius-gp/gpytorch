#!/usr/bin/env python3

import torch

from .. import settings


def minres(matmul_closure, rhs, eps=1e-25, shifts=None, value=None, max_iter=None):
    r"""
    Perform MINRES to find solutions to :math:`(\mathbf K + \alpha \sigma \mathbf I) \mathbf x = \mathbf b`.
    Will find solutions for multiple shifts :math:`\sigma` at the same time.

    :param callable matmul_closure: Function to perform matmul with.
    :param torch.Tensor rhs: The vector :math:`\mathbf b` to solve against.
    :param torch.Tensor shifts: (default None) The shift :math:`\sigma` values. If set to None,
        then :math:`\sigma=0`.
    :param float value: (default None) The multiplicative constant :math:`\alpha`. If set to None,
        then :math:`\alpha=0`.
    :param int max_iter: (default None) The maximum number of minres iterations. If set to None, then
        uses the constant stored in :obj:`gpytorch.settings.max_cg_iterations`.
    :rtype: torch.Tensor
    :return: The solves :math:`\mathbf x`. The shape will correspond to the size of `rhs` and `shifts`.
    """
    # Default values
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    mm_ = matmul_closure

    if shifts is None:
        shifts = torch.tensor(0.0, dtype=rhs.dtype, device=rhs.device)

    # Scale the rhs
    squeeze = False
    if rhs.dim() == 1:
        rhs = rhs.unsqueeze(-1)
        squeeze = True

    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(1e-10)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)
    rhs = rhs.div(rhs_norm)

    # Use the right number of iterations
    if max_iter is None:
        max_iter = settings.max_cg_iterations.value()
    max_iter = min(max_iter, rhs.size(-2) + 1)

    # Epsilon (to prevent nans)
    eps = torch.tensor(eps, dtype=rhs.dtype, device=rhs.device)

    # Create space for matmul product, solution
    prod = mm_(rhs)
    if value is not None:
        prod.mul_(value)

    solution = torch.zeros(shifts.shape + prod.shape, dtype=rhs.dtype, device=rhs.device)

    # Reisze shifts

    shifts = shifts.view(shifts.shape + torch.Size([1 for _ in prod.shape]))

    # Variasbles for Lanczos terms
    alpha_curr = torch.empty(prod.shape[:-2] + (1, prod.size(-1)), dtype=rhs.dtype, device=rhs.device)
    alpha_shifted_curr = torch.empty(solution.shape[:-2] + (1, prod.size(-1)), dtype=rhs.dtype, device=rhs.device)
    beta_prev = rhs_norm.clone().expand((prod.shape[:-2] + (1, prod.size(-1)))).contiguous()
    beta_curr = torch.empty_like(beta_prev)
    qvec_prev2 = torch.zeros_like(prod)
    qvec_prev1 = rhs.clone().expand_as(prod).contiguous()
    qvec_curr = torch.empty_like(qvec_prev2)

    # Variables for the QR rotation
    # 1) Components of the Givens rotations
    cos_prev2 = torch.ones(solution.shape[:-2] + (1, rhs.size(-1)), dtype=rhs.dtype, device=rhs.device)
    sin_prev2 = torch.zeros(solution.shape[:-2] + (1, rhs.size(-1)), dtype=rhs.dtype, device=rhs.device)
    cos_prev1 = torch.ones_like(cos_prev2)
    sin_prev1 = torch.zeros_like(sin_prev2)
    radius_curr = torch.empty_like(cos_prev1)
    cos_curr = torch.empty_like(cos_prev1)
    sin_curr = torch.empty_like(cos_prev1)
    # 2) Terms QR decomposition of T
    subsub_diag_term = torch.empty_like(alpha_shifted_curr)
    sub_diag_term = torch.empty_like(alpha_shifted_curr)
    diag_term = torch.empty_like(alpha_shifted_curr)

    # Variables for the solution updates
    # 1) The "search" vectors of the solution
    # Equivalent to the vectors of Q R^{-1}, where Q is the matrix of Lanczos vectors and
    # R is the QR factor of the tridiagonal Lanczos matrix.
    search_prev2 = torch.zeros_like(solution)
    search_prev1 = torch.zeros_like(solution)
    search_curr = torch.empty_like(search_prev1)
    # 2) The "scaling" terms of the search vectors
    # Equivalent to the terms of V^T Q^T rhs, where Q is the matrix of Lanczos vectors and
    # V is the QR orthonormal of the tridiagonal Lanczos matrix.
    scale_prev = beta_prev.repeat(shifts.shape)
    scale_curr = torch.empty_like(scale_prev)

    # Perform iterations
    for i in range(max_iter):
        # Perform matmul
        prod = mm_(qvec_prev1)
        if value is not None:
            prod.mul_(value)

        # Get next Lanczos terms
        # --> alpha_curr, beta_curr, qvec_curr
        qvec_curr.copy_(prod)
        torch.sum(prod.mul_(qvec_prev1), -2, keepdim=True, out=alpha_curr)

        qvec_curr.addcmul_(alpha_curr, qvec_prev1, value=-1).addcmul_(beta_prev, qvec_prev2, value=-1)

        torch.mul(sin_prev2, beta_prev, out=subsub_diag_term)
        torch.mul(cos_prev2, beta_prev, out=sub_diag_term)

        torch.norm(qvec_curr, p=2, dim=-2, keepdim=True, out=beta_curr)
        beta_curr.clamp_min_(eps)

        qvec_curr.div_(beta_curr)

        torch.add(alpha_curr, shifts, out=alpha_shifted_curr)

        torch.mul(alpha_shifted_curr, cos_prev1, out=diag_term).addcmul_(sin_prev1, sub_diag_term, value=-1)
        sub_diag_term.mul_(cos_prev1).addcmul_(sin_prev1, alpha_shifted_curr)
        # 3) Compute next Givens terms
        torch.mul(diag_term, diag_term, out=radius_curr).addcmul_(beta_curr, beta_curr).sqrt_()
        cos_curr = torch.div(diag_term, radius_curr)
        sin_curr = torch.div(beta_curr, radius_curr)
        # 4) Apply current Givens rotation
        diag_term.mul_(cos_curr).addcmul_(sin_curr, beta_curr)

        # Update the solution
        # --> search_curr, scale_curr solution
        # 1) Apply the latest Givens rotation to the Lanczos-rhs ( ||rhs|| e_1 )
        # This is getting the scale terms for the "search" vectors
        scale_curr = torch.mul(scale_prev, sin_curr).mul_(-1)
        scale_prev.mul_(cos_curr)
        # 2) Get the new search vector
        # from IPython.core.debugger import set_trace
        # set_trace()
        search_curr = torch.addcmul(qvec_prev1, sub_diag_term, search_prev1, value=-1)
        search_curr.addcmul_(subsub_diag_term, search_prev2, value=-1)
        search_curr.div_(diag_term)
        # 3) Update the solution
        solution.addcmul_(scale_prev, search_curr)

        # Update terms for next iteration
        # Lanczos terms

        qvec_prev2.copy_(qvec_prev1)
        qvec_prev1.copy_(qvec_curr)
        # Givens rotations terms
        cos_prev2.copy_(cos_prev1)
        sin_prev2.copy_(sin_prev1)
        search_prev2.copy_(search_prev1)

        beta_prev = beta_curr
        cos_prev1 = cos_curr
        sin_prev1 = sin_curr
        # Search vector terms)
        scale_prev = scale_curr
        search_prev1 = search_curr
        # For rhs-s that are close to zero, set them to zero
        solution.masked_fill_(rhs_is_zero, 0)

    # Store the number of iterations taken
    settings.record_ciq_stats.num_minres_iter = i + 1

    if squeeze:
        solution = solution.squeeze(-1)

    return solution
