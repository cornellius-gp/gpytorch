#!/usr/bin/env python3

import torch
from .. import settings


def _default_preconditioner(x):
    return x.clone()


def linear_cg(
    matmul_closure,
    rhs,
    n_tridiag=0,
    tolerance=1e-6,
    eps=1e-10,
    max_iter=None,
    max_tridiag_iter=None,
    initial_guess=None,
    preconditioner=None,
):
    """
    Implements the linear conjugate gradients method for (approximately) solving systems of the form

        lhs result = rhs

    for positive definite and symmetric matrices.

    Args:
      - matmul_closure - a function which performs a left matrix multiplication with lhs_mat
      - rhs - the right-hand side of the equation
      - n_tridiag - returns a tridiagonalization of the first n_tridiag columns of rhs
      - tolerance - stop the solve when the max residual is less than this
      - eps - noise to add to prevent division by zero
      - max_iter - the maximum number of CG iterations
      - max_tridiag_iter - the maximum size of the tridiagonalization matrix
      - initial_guess - an initial guess at the solution `result`
      - precondition_closure - a functions which left-preconditions a supplied vector

    Returns:
      result - a solution to the system (if n_tridiag is 0)
      result, tridiags - a solution to the system, and corresponding tridiagonal matrices (if n_tridiag > 0)
    """
    # Unsqueeze, if necesasry
    is_vector = rhs.ndimension() == 1
    if is_vector:
        rhs = rhs.unsqueeze(-1)

    # Some default arguments
    if max_iter is None:
        max_iter = settings.max_cg_iterations.value()
    if max_tridiag_iter is None:
        max_tridiag_iter = settings.max_lanczos_quadrature_iterations.value()
    if initial_guess is None:
        initial_guess = torch.zeros_like(rhs)
    if preconditioner is None:
        preconditioner = _default_preconditioner

    # If we are running m CG iterations, we obviously can't get more than m Lanczos coefficients
    if max_tridiag_iter > max_iter:
        raise RuntimeError("Getting a tridiagonalization larger than the number of CG iterations run is not possible!")

    # Check matmul_closure object
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    elif not callable(matmul_closure):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    # Get some constants
    batch_shape = rhs.shape[:-2]
    num_rows = rhs.size(-2)
    n_iter = min(max_iter, num_rows) if settings.terminate_cg_by_size.on() else max_iter
    n_tridiag_iter = min(max_tridiag_iter, num_rows)

    # result <- x_{0}
    result = initial_guess

    # residual: residual_{0} = b_vec - lhs x_{0}
    residual = rhs - matmul_closure(result)

    # Check for NaNs
    if not torch.equal(residual, residual):
        raise RuntimeError("NaNs encounterd when trying to perform matrix-vector multiplication")

    # Sometime we're lucky and the preconditioner solves the system right away
    residual_norm = residual.norm(2, dim=-2)
    if (residual_norm < tolerance).all() and not n_tridiag:
        n_iter = 0  # Skip the iteration!

    # Otherwise, let's define precond_residual and curr_conjugate_vec
    else:
        # precon_residual{0} = M^-1 residual_{0}
        precond_residual = preconditioner(residual)
        curr_conjugate_vec = precond_residual
        residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

        # Define storage matrices
        mul_storage = torch.empty_like(residual)
        alpha = torch.empty(*batch_shape, rhs.size(-1), dtype=residual.dtype, device=residual.device)
        beta = torch.empty_like(alpha)

    # Define tridiagonal matrices, if applicable
    if n_tridiag:
        t_mat = torch.zeros(
            n_tridiag_iter, n_tridiag_iter, *batch_shape, n_tridiag, dtype=alpha.dtype, device=alpha.device
        )
        alpha_reciprocal = torch.empty(*batch_shape, n_tridiag, dtype=t_mat.dtype, device=t_mat.device)
        prev_alpha_reciprocal = torch.empty_like(alpha_reciprocal)
        prev_beta = torch.empty_like(alpha_reciprocal)

    update_tridiag = True
    last_tridiag_iter = 0
    # Start the iteration
    for k in range(n_iter):
        # Get next alpha
        # alpha_{k} = (residual_{k-1}^T precon_residual{k-1}) / (p_vec_{k-1}^T mat p_vec_{k-1})
        mvms = matmul_closure(curr_conjugate_vec)
        torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
        torch.sum(mul_storage, -2, keepdim=True, out=alpha)
        alpha.add_(eps)
        torch.div(residual_inner_prod, alpha, out=alpha)

        # Update result
        # result_{k} = result_{k-1} + alpha_{k} p_vec_{k-1}
        torch.addcmul(result, alpha, curr_conjugate_vec, out=result)

        # Update residual
        # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
        torch.addcmul(residual, -1, alpha, mvms, out=residual)

        # If residual are sufficiently small, then exit loop
        # Alternatively, exit if this is our last iteration
        torch.norm(residual, 2, dim=-2, out=residual_norm)
        if (residual_norm < tolerance).all() and not (n_tridiag and k < n_tridiag_iter):
            break

        # Update precond_residual
        # precon_residual{k} = M^-1 residual_{k}
        precond_residual = preconditioner(residual)

        # beta_{k} = (precon_residual{k}^T r_vec_{k}) / (precon_residual{k-1}^T r_vec_{k-1})
        residual_inner_prod.add_(eps)
        torch.reciprocal(residual_inner_prod, out=beta)
        torch.mul(residual, precond_residual, out=mul_storage)
        torch.sum(mul_storage, -2, keepdim=True, out=residual_inner_prod)
        beta.mul_(residual_inner_prod)

        # Update curr_conjugate_vec
        # curr_conjugate_vec_{k} = precon_residual{k} + beta_{k} curr_conjugate_vec_{k-1}
        curr_conjugate_vec.mul_(beta).add_(precond_residual)

        # Update tridiagonal matrices, if applicable
        if n_tridiag and k < n_tridiag_iter and update_tridiag:
            alpha_tridiag = alpha.squeeze_(-2).narrow(-1, 0, n_tridiag)
            beta_tridiag = beta.squeeze_(-2).narrow(-1, 0, n_tridiag)
            torch.reciprocal(alpha_tridiag, out=alpha_reciprocal)

            if k == 0:
                t_mat[k, k].copy_(alpha_reciprocal)
            else:
                torch.addcmul(alpha_reciprocal, prev_beta, prev_alpha_reciprocal, out=t_mat[k, k])
                torch.mul(prev_beta.sqrt_(), prev_alpha_reciprocal, out=t_mat[k, k - 1])
                t_mat[k - 1, k].copy_(t_mat[k, k - 1])

                if t_mat[k - 1, k].max() < 1e-6:
                    update_tridiag = False

            last_tridiag_iter = k

            prev_alpha_reciprocal.copy_(alpha_reciprocal)
            prev_beta.copy_(beta_tridiag)

    if is_vector:
        result = result.squeeze(-1)

    if n_tridiag:
        t_mat = t_mat[: last_tridiag_iter + 1, : last_tridiag_iter + 1]
        return result, t_mat.permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous()
    else:
        return result
