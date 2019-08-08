#!/usr/bin/env python3

import torch


def minres(matmul_closure, rhs, shifts=None, max_num_iter=100):
    # Default values
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    if shifts is None:
        shifts = torch.tensor(0., dtype=rhs.dtype, device=rhs.device)

    # Scale the rhs
    squeeze = False
    if rhs.dim() == 1:
        rhs = rhs.unsqueeze(-1)
        squeeze = True
    initial_norm = rhs.norm(dim=-2, keepdim=True)
    rhs = rhs.div(initial_norm)

    # Create space for matmul product, solution
    prod = matmul_closure(rhs)
    solution = torch.zeros(*shifts.shape, *prod.shape, dtype=rhs.dtype, device=rhs.device)

    # Reisze shifts
    shifts = shifts.view(*shifts.shape, *[1 for _ in prod.shape])

    # Variasbles for Lanczos terms
    alpha_curr = None
    alpha_shifted_curr = None
    beta_prev = initial_norm.expand(*prod.shape[:-2], 1, prod.size(-1)).contiguous()
    beta_curr = None
    qvec_prev2 = torch.zeros_like(prod)
    qvec_prev1 = rhs.expand_as(prod).contiguous()
    qvec_curr = None

    # Variables for the QR rotation
    # 1) Components of the Givens rotations
    cos_prev2 = torch.ones(*solution.shape[:-2], 1, rhs.size(-1), dtype=rhs.dtype, device=rhs.device)
    sin_prev2 = torch.zeros(*solution.shape[:-2], 1, rhs.size(-1), dtype=rhs.dtype, device=rhs.device)
    cos_prev1 = torch.ones_like(cos_prev2)
    sin_prev1 = torch.zeros_like(sin_prev2)
    radius_curr = None
    cos_curr = None
    sin_curr = None
    # 2) Terms QR decomposition of T
    subsub_diag_term = None
    sub_diag_term = None
    diag_term = None

    # Variables for the solution updates
    # 1) The "search" vectors of the solution
    # Equivalent to the vectors of Q R^{-1}, where Q is the matrix of Lanczos vectors and
    # R is the QR factor of the tridiagonal Lanczos matrix.
    search_prev2 = torch.zeros_like(prod)
    search_prev1 = torch.zeros_like(prod)
    search_curr = None
    # 2) The "scaling" terms of the search vectors
    # Equivalent to the terms of V^T Q^T rhs, where Q is the matrix of Lanczos vectors and
    # V is the QR orthonormal of the tridiagonal Lanczos matrix.
    scale_prev = beta_prev.clone()
    scale_curr = None

    # Perform iterations
    num_iter = min(rhs.size(-2), max_num_iter)
    for i in range(num_iter):
        # Perform matmul
        prod = matmul_closure(qvec_prev1)

        # Get next Lanczos terms
        # alpha_curr, beta_curr, qvec_curr
        alpha_curr = torch.mul(qvec_prev1, prod).sum(-2, keepdim=True)
        qvec_curr = prod.addcmul_(-1, alpha_curr, qvec_prev1).addcmul_(-1, beta_prev, qvec_prev2)
        beta_curr = qvec_curr.norm(dim=-2, keepdim=True)
        qvec_curr.div_(beta_curr)

        # Get shifted alpha
        alpha_shifted_curr = torch.add(alpha_curr, shifts)

        # Perfom next step of the QR factorization
        # 1) Apply second previous Givens rotation
        subsub_diag_term = sin_prev2 * beta_prev
        sub_diag_term = cos_prev2 * beta_prev
        # 2) Apply previous Givens rotation
        diag_term = torch.mul(alpha_shifted_curr, cos_prev1).addcmul_(-1, sin_prev1, sub_diag_term)
        sub_diag_term.mul_(cos_prev1).addcmul_(sin_prev1, alpha_shifted_curr)
        # 3) Compute next Givens terms
        radius_curr = torch.add(diag_term.pow(2), beta_curr.pow(2)).sqrt_()
        cos_curr = diag_term / radius_curr
        sin_curr = beta_curr / radius_curr
        # 4) Apply current Givens rotation
        diag_term.mul_(cos_curr).addcmul_(sin_curr, beta_curr)

        # Update the solution
        # 1) Apply the V matrix of QR to the Lanczos-rhs (= Q^T rhs)
        # This is getting the scale terms for the "search" vectors
        scale_curr = scale_prev.mul(sin_curr).mul_(-1)
        scale_prev = scale_prev.mul(cos_curr)
        # 2) Get the new search vector
        search_curr = (
            qvec_prev1.addcmul(-1, sub_diag_term, search_prev1).
            addcmul_(-1, subsub_diag_term, search_prev2).div_(diag_term)
        )
        # 3) Update the solution
        solution.addcmul_(scale_prev, search_curr)

        # Update terms for next iteration
        # Lanczos terms
        beta_prev = beta_curr
        qvec_prev2 = qvec_prev1
        qvec_prev1 = qvec_curr
        # Givens rotations terms
        cos_prev2 = cos_prev1
        sin_prev2 = sin_prev1
        cos_prev1 = cos_curr
        sin_prev1 = sin_curr
        # Search vector terms
        scale_prev = scale_curr
        search_prev2 = search_prev1
        search_prev1 = search_curr

    if squeeze:
        solution = solution.squeeze(-1)
    return solution
