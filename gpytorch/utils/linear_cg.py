from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .. import settings


def _default_preconditioner(x):
    return x.clone()


def linear_cg(
    matmul_closure,
    rhs,
    n_tridiag=0,
    tolerance=1e-6,
    eps=1e-20,
    max_iter=None,
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
    if initial_guess is None:
        initial_guess = rhs.new(rhs.size()).zero_()
    if preconditioner is None:
        preconditioner = _default_preconditioner

    # Check matmul_closure object
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    elif not callable(matmul_closure):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    # Get some constants
    n_rows = rhs.size(-2)
    n_iter = min(max_iter, n_rows)

    # result <- x_{0}
    result = initial_guess

    # residual: residual_{0} = b_vec - lhs x_{0}
    residual = rhs - matmul_closure(result)

    # Check for NaNs
    if not torch.equal(residual, residual):
        raise RuntimeError(
            "NaNs encounterd when trying to perform matrix-vector multiplication"
        )

    # Sometime we're lucky and the preconditioner solves the system right away
    residual_norm = residual.norm(2, dim=-2)
    if not torch.sum(residual_norm > tolerance) and not n_tridiag:
        n_iter = 0  # Skip the iteration!

    # Otherwise, let's define precond_residual and curr_conjugate_vec
    else:
        # precon_residual{0} = M^-1 residual_{0}
        precond_residual = preconditioner(residual)
        curr_conjugate_vec = precond_residual
        residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

        # Define storage matrices
        mul_storage = residual.new(residual.size())
        alpha = residual.new(
            rhs.size(0), 1, rhs.size(-1)
        ) if rhs.ndimension() == 3 else residual.new(
            1, rhs.size(-1)
        )
        beta = alpha.new(alpha.size())

    # Define tridiagonal matrices, if applicable
    if n_tridiag:
        if rhs.ndimension() == 3:
            t_mat = residual.new(n_iter, n_iter, rhs.size(0), n_tridiag).zero_()
            alpha_reciprocal = alpha.new(rhs.size(0), n_tridiag)
        else:
            t_mat = residual.new(n_iter, n_iter, n_tridiag).zero_()
            alpha_reciprocal = alpha.new(n_tridiag)

        prev_alpha_reciprocal = alpha.new(alpha_reciprocal.size())
        prev_beta = alpha.new(alpha_reciprocal.size())

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
        if not (torch.sum(residual_norm > tolerance)) and not n_tridiag:
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
        if n_tridiag:
            alpha_tridiag = alpha.squeeze_(-2).narrow(-1, 0, n_tridiag)
            beta_tridiag = beta.squeeze_(-2).narrow(-1, 0, n_tridiag)
            torch.reciprocal(alpha_tridiag, out=alpha_reciprocal)

            if k == 0:
                t_mat[k, k].copy_(alpha_reciprocal)
            else:
                torch.addcmul(
                    alpha_reciprocal, prev_beta, prev_alpha_reciprocal, out=t_mat[k, k]
                )
                torch.mul(prev_beta.sqrt_(), prev_alpha_reciprocal, out=t_mat[k, k - 1])
                t_mat[k - 1, k].copy_(t_mat[k, k - 1])

            prev_alpha_reciprocal.copy_(alpha_reciprocal)
            prev_beta.copy_(beta_tridiag)

    if is_vector:
        result = result.squeeze(-1)

    if n_tridiag:
        if rhs.ndimension() == 3:
            return result, t_mat.permute(3, 2, 0, 1).contiguous()
        else:
            return result, t_mat.permute(2, 0, 1).contiguous()
    else:
        return result
