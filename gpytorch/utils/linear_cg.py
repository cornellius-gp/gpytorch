import torch
import gpytorch


def linear_cg(matmul_closure, rhs, tolerance=1e-6, eps=1e-6, max_iter=None,
              initial_guess=None, preconditioner=None):
    """
    Implements the linear conjugate gradients method for (approximately) solving systems of the form

        lhs result = rhs

    for positive definite and symmetric matrices.

    Args:
      - matmul_closure - a function which performs a left matrix multiplication with lhs_mat
      - rhs - the right-hand side of the equation
      - tolerance - stop the solve when the max residual is less than this
      - eps - noise to add to prevent division by zero
      - max_iter - the maximum number of CG iterations
      - initial_guess - an initial guess at the solution `result`
      - precondition_closure - a functions which left-preconditions a supplied vector

    Returns:
      result - a solution to the system
    """
    # Unsqueeze, if necesasry
    isvector = rhs.ndimension() == 1
    if isvector:
        rhs = rhs.unsqueeze(-1)

    # Some default arguments
    if max_iter is None:
        max_iter = gpytorch.functions.max_cg_iterations
    if initial_guess is None:
        initial_guess = rhs.new(rhs.size()).zero_()
    if preconditioner is None:
        def _preconditioner(mat):
            return mat
        preconditioner = _preconditioner

    # Check matmul_closure object
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    elif not callable(matmul_closure):
        raise RuntimeError('matmul_closure must be a tensor, or a callable object!')

    # Get some constants
    n_rows = rhs.size(-2)
    n_iter = min(max_iter, n_rows)

    # result <- x_{0}
    result = initial_guess

    # residual: residual_{0} = b_vec - lhs x_{0}
    residual = rhs - matmul_closure(result)

    # Sometime we're lucky and the preconditioner solves the system right away
    residual_norm = residual.norm(2, dim=-2)
    if not torch.sum(residual_norm > tolerance):
        n_iter = 0  # Skip the iteration!

    # Otherwise, let's define z_vec and p_vec
    else:
        # z_vec_{0} = M^-1 residual_{0}
        z_vec = preconditioner(residual)
        p_vec = z_vec
        residual_dot_z_vec = residual.mul(z_vec).sum(-2, keepdim=True)

    # Start the iteration
    for k in range(n_iter):
        # Store previous values
        prev_residual_dot_z_vec = residual_dot_z_vec

        # Get next alpha
        # alpha_{k} = (residual_{k-1}^T z_vec_{k-1}) / (p_vec_{k-1}^T lhs p_vec_{k-1})
        lhs_p_vec = matmul_closure(p_vec)
        lhs_p_vec_quad_form = p_vec.mul(lhs_p_vec).sum(-2, keepdim=True)
        alpha = residual_dot_z_vec.div(lhs_p_vec_quad_form + eps)

        # Update result
        # result_{k} = result_{k-1} + alpha_{k} p_vec_{k-1}
        result = result + alpha.mul(p_vec)

        # Update residual
        # residual_{k} = residual_{k-1} - alpha_{k} lhs p_vec_{k-1}
        residual = residual - lhs_p_vec.mul(alpha)

        # If residual are sufficiently small, then exit loop
        # Alternatively, exit if this is our last iteration
        residual_norm = residual.norm(2, dim=-2)
        if not (torch.sum(residual_norm > tolerance) and (k + 1) < n_iter):
            break

        # Update z_vec
        # z_vec_{k} = M^-1 residual_{k}
        z_vec = preconditioner(residual)

        # beta_{k} = (z_vec_{k}^T r_vec_{k}) / (z_vec_{k-1}^T r_vec_{k-1})
        residual_dot_z_vec = residual.mul(z_vec).sum(-2, keepdim=True)
        beta = residual_dot_z_vec.div(prev_residual_dot_z_vec + eps)

        # Update p_vec
        # p_vec_{k} = z_vec_{k} + beta_{k} p_vec_{k-1}
        p_vec = z_vec + p_vec.mul(beta)

    if isvector:
        result = result.squeeze(-1)
    return result
