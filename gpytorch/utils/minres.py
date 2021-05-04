#!/usr/bin/env python3

import torch

from .. import settings
from .broadcasting import _pad_with_singletons


def minres(matmul_closure, rhs, eps=1e-25, shifts=None, value=None, max_iter=None, preconditioner=None):
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
    if preconditioner is None:
        preconditioner = lambda x: x.clone()

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

    # Resize shifts
    shifts = _pad_with_singletons(shifts, 0, prod.dim() - shifts.dim() + 1)
    solution = torch.zeros(shifts.shape[:1] + prod.shape, dtype=rhs.dtype, device=rhs.device)

    # Variables for Lanczos terms
    zvec_prev2 = torch.zeros_like(prod)
    zvec_prev1 = rhs.clone().expand_as(prod).contiguous()
    qvec_prev1 = preconditioner(zvec_prev1)
    alpha_curr = torch.empty(prod.shape[:-2] + (1, prod.size(-1)), dtype=rhs.dtype, device=rhs.device)
    alpha_shifted_curr = torch.empty(solution.shape[:-2] + (1, prod.size(-1)), dtype=rhs.dtype, device=rhs.device)
    beta_prev = (zvec_prev1 * qvec_prev1).sum(dim=-2, keepdim=True).sqrt_()
    beta_curr = torch.empty_like(beta_prev)
    tmpvec = torch.empty_like(qvec_prev1)

    # Divide by beta_prev
    zvec_prev1.div_(beta_prev)
    qvec_prev1.div_(beta_prev)

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
    search_update = torch.empty_like(search_prev1)
    # 2) The "scaling" terms of the search vectors
    # Equivalent to the terms of V^T Q^T rhs, where Q is the matrix of Lanczos vectors and
    # V is the QR orthonormal of the tridiagonal Lanczos matrix.
    scale_prev = beta_prev.repeat(shifts.size(0), *([1] * beta_prev.dim()))
    scale_curr = torch.empty_like(scale_prev)

    # Terms for checking for convergence
    solution_norm = torch.zeros(*solution.shape[:-2], solution.size(-1), dtype=solution.dtype, device=solution.device)
    search_update_norm = torch.zeros_like(solution_norm)

    # Maybe log
    if settings.verbose_linalg.on():
        settings.verbose_linalg.logger.debug(
            f"Running MINRES on a {rhs.shape} RHS for {max_iter} iterations (tol={settings.minres_tolerance.value()}). "
            f"Output: {solution.shape}."
        )

    # Perform iterations
    for i in range(max_iter + 2):
        # Perform matmul
        prod = mm_(qvec_prev1)
        if value is not None:
            prod.mul_(value)

        # Get next Lanczos terms
        # --> alpha_curr, beta_curr, qvec_curr
        torch.mul(prod, qvec_prev1, out=tmpvec)
        torch.sum(tmpvec, -2, keepdim=True, out=alpha_curr)

        zvec_curr = prod.addcmul_(alpha_curr, zvec_prev1, value=-1).addcmul_(beta_prev, zvec_prev2, value=-1)

        qvec_curr = preconditioner(zvec_curr)
        torch.mul(zvec_curr, qvec_curr, out=tmpvec)
        torch.sum(tmpvec, -2, keepdim=True, out=beta_curr)
        beta_curr.sqrt_()
        beta_curr.clamp_min_(eps)

        zvec_curr.div_(beta_curr)
        qvec_curr.div_(beta_curr)

        # Perform JIT-ted update
        conv = _jit_minres_updates(
            solution,
            shifts,
            eps,
            qvec_prev1,
            alpha_curr,
            alpha_shifted_curr,
            beta_prev,
            beta_curr,
            cos_prev2,
            cos_prev1,
            cos_curr,
            sin_prev2,
            sin_prev1,
            sin_curr,
            radius_curr,
            subsub_diag_term,
            sub_diag_term,
            diag_term,
            search_prev2,
            search_prev1,
            search_curr,
            search_update,
            scale_prev,
            scale_curr,
            search_update_norm,
            solution_norm,
        )

        # Check convergence criterion
        if (i + 1) % 10 == 0:
            torch.norm(search_update, dim=-2, out=search_update_norm)
            torch.norm(solution, dim=-2, out=solution_norm)
            conv = search_update_norm.div_(solution_norm).mean().item()
            if conv < settings.minres_tolerance.value():
                break

        # Update terms for next iteration
        # Lanczos terms
        zvec_prev2, zvec_prev1 = zvec_prev1, prod
        qvec_prev1 = qvec_curr
        beta_prev, beta_curr = beta_curr, beta_prev
        # Givens rotations terms
        cos_prev2, cos_prev1, cos_curr = cos_prev1, cos_curr, cos_prev2
        sin_prev2, sin_prev1, sin_curr = sin_prev1, sin_curr, sin_prev2
        # Search vector terms)
        search_prev2, search_prev1, search_curr = search_prev1, search_curr, search_prev2
        scale_prev, scale_curr = scale_curr, scale_prev

    # For rhs-s that are close to zero, set them to zero
    solution.masked_fill_(rhs_is_zero, 0)

    if squeeze:
        solution = solution.squeeze(-1)
        rhs = rhs.squeeze(-1)
        rhs_norm = rhs_norm.squeeze(-1)

    if shifts.numel() == 1:
        # If we weren't shifting we shouldn't return a batch output
        solution = solution.squeeze(0)

    return solution.mul_(rhs_norm)


def _jit_minres_updates(
    solution,
    shifts,
    eps,
    qvec_prev1,
    alpha_curr,
    alpha_shifted_curr,
    beta_prev,
    beta_curr,
    cos_prev2,
    cos_prev1,
    cos_curr,
    sin_prev2,
    sin_prev1,
    sin_curr,
    radius_curr,
    subsub_diag_term,
    sub_diag_term,
    diag_term,
    search_prev2,
    search_prev1,
    search_curr,
    search_update,
    scale_prev,
    scale_curr,
    search_update_norm,
    solution_norm,
):
    # Start givens rotation
    # Givens rotation from 2 steps ago
    torch.mul(sin_prev2, beta_prev, out=subsub_diag_term)
    torch.mul(cos_prev2, beta_prev, out=sub_diag_term)

    # Compute shifted alpha
    torch.add(alpha_curr, shifts, out=alpha_shifted_curr)

    # Givens rotation from 1 step ago
    torch.mul(alpha_shifted_curr, cos_prev1, out=diag_term).addcmul_(sin_prev1, sub_diag_term, value=-1)
    sub_diag_term.mul_(cos_prev1).addcmul_(sin_prev1, alpha_shifted_curr)

    # 3) Compute next Givens terms
    torch.mul(diag_term, diag_term, out=radius_curr).addcmul_(beta_curr, beta_curr).sqrt_()
    cos_curr = torch.div(diag_term, radius_curr, out=cos_curr)
    sin_curr = torch.div(beta_curr, radius_curr, out=sin_curr)
    # 4) Apply current Givens rotation
    diag_term.mul_(cos_curr).addcmul_(sin_curr, beta_curr)

    # Update the solution
    # --> search_curr, scale_curr solution
    # 1) Apply the latest Givens rotation to the Lanczos-rhs ( ||rhs|| e_1 )
    # This is getting the scale terms for the "search" vectors
    torch.mul(scale_prev, sin_curr, out=scale_curr).mul_(-1)
    scale_prev.mul_(cos_curr)
    # 2) Get the new search vector
    torch.addcmul(qvec_prev1, sub_diag_term, search_prev1, value=-1, out=search_curr)
    search_curr.addcmul_(subsub_diag_term, search_prev2, value=-1)
    search_curr.div_(diag_term)

    # 3) Update the solution
    torch.mul(search_curr, scale_prev, out=search_update)
    solution.add_(search_update)
