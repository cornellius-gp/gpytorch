import torch
from .. import settings


def _default_preconditioner(x):
    return x.clone()


def linear_log_cg_re(
    matmul_closure,
    rhs,
    max_iter,
    initial_guess=None,
    preconditioner=None,
    tolerance=None,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_tridiag_iter=0,
    n_tridiag=0,
):
    if preconditioner is None:
        preconditioner = _default_preconditioner
    if tolerance is None:
        if settings._use_eval_tolerance.on():
            tolerance = settings.eval_cg_tolerance.value()
        else:
            tolerance = settings.cg_tolerance.value()
    if initial_guess is None:
        initial_guess = torch.zeros_like(rhs)
    x0 = initial_guess
    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(eps)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)
    rhs = rhs.div(rhs_norm)

    state = initialize_log_re(matmul_closure, rhs, preconditioner, x0, max_iter)
    for k in range(max_iter):
        state = take_cg_step_log_re(state, matmul_closure, preconditioner)
        if cond_fun(state, tolerance, max_iter):
            break

    x0 = state[0]
    x0 = x0.mul(rhs_norm)
    if n_tridiag > 0:
        return x0, None
    else:
        return x0


def initialize_log_re(A, b, preconditioner, x0, max_iters):
    r0 = b - A(x0)
    z0 = preconditioner(r0)
    p0 = z0
    log_gamma0 = update_log_gamma_unclipped(r=r0, z=z0)
    u_all = torch.zeros(size=(max_iters,) + b.shape, dtype=x0.dtype, device=x0.device)
    return (x0, r0, log_gamma0, p0, u_all, torch.tensor(0, dtype=torch.int32))


def take_cg_step_log_re(state, A, preconditioner):
    x0, r0, log_gamma0, p0, u_all, k = state
    r_norm = torch.linalg.norm(r0, axis=-2, keepdim=True)
    has_converged = r_norm < torch.tensor(1.e-6, dtype=p0.dtype)
    Ap0 = A(p0)

    alpha = update_alpha_log_unclipped(log_gamma0, p0, Ap0, has_converged)
    x1 = x0 + alpha * p0
    r1 = r0 - alpha * Ap0
    for i in range(k - 1):
        dotprod = torch.sum(r1 * u_all[i], dim=-2, keepdim=True) * u_all[i]
        r1 = torch.where(has_converged, r1, r1 - dotprod)
    z1 = preconditioner(r1)
    log_gamma1, beta = update_log_gamma_beta_unclipped(
        r1, z1, log_gamma0, has_converged)
    u_all[k] = r1 / torch.sqrt(torch.exp(log_gamma1))
    p1 = z1 + beta * p0
    # print_progress(k, alpha, r1, torch.exp(log_gamma1), beta)

    return (x1, r1, log_gamma1, p1, u_all, k + 1)


def update_alpha_log_unclipped(log_gamma, p, Ap, has_converged):
    log_alpha_abs, sign = compute_robust_denom_unclipped(p, Ap)
    log_denom = logsumexp(tensor=log_alpha_abs, dim=-2, mask=sign)
    alpha = torch.exp(log_gamma - log_denom)
    alpha = torch.where(has_converged, torch.zeros_like(alpha), alpha)
    return alpha


def compute_robust_denom_unclipped(p, Ap):
    p_abs = torch.clip(torch.abs(p), min=1.e-8)
    Ap_abs = torch.clip(torch.abs(Ap), min=1.e-8)
    sign = torch.sign(p) * torch.sign(Ap)
    log_alpha_abs = torch.log(p_abs) + torch.log(Ap_abs)
    return log_alpha_abs, sign


def update_log_gamma_beta_unclipped(r, z, log_gamma0, has_converged):
    log_gamma1 = update_log_gamma_unclipped(r, z)
    beta = torch.exp(log_gamma1 - log_gamma0)
    beta = torch.where(has_converged, torch.zeros_like(beta), beta)
    return log_gamma1, beta


def update_log_gamma_unclipped(r, z):
    r_abs = torch.abs(r)
    z_abs = torch.abs(z)
    sign = torch.sign(r) * torch.sign(z)
    log_gamma_abs = torch.log(r_abs) + torch.log(z_abs)
    log_gamma = logsumexp(tensor=log_gamma_abs, dim=-2, mask=sign)
    return log_gamma


def cond_fun(state, tolerance, max_iters):
    _, r, *_, k = state
    rs = torch.linalg.norm(r, axis=-2)
    res_meet = torch.mean(rs) < tolerance
    min_val = torch.minimum(torch.tensor(10, dtype=torch.int32),
                            torch.tensor(max_iters, dtype=torch.int32))
    flag = ((res_meet) & (k >= min_val) | (k > max_iters))
    return flag


def logsumexp(tensor, dim=-1, mask=None):
    max_entry = torch.max(tensor, dim, keepdim=True)[0]
    summ = torch.sum((tensor - max_entry).exp() * mask, dim, keepdim=True)
    return max_entry + summ.log()


def print_progress(k, alpha, r1, gamma1, beta):
    print('\n===================================================')
    print(f'Iter {k}')
    print(f'Residual norm mean: {torch.mean(torch.linalg.norm(r1, axis=0))}')
    print(f'Residual norm max: {torch.max(torch.linalg.norm(r1, axis=0))}')
    print(f'Residual norm: {torch.linalg.norm(r1, axis=0)}')
    print('alpha')
    print(alpha)
    print(f'Alpha mean: {torch.mean(alpha)}')
    print('gamma')
    print(gamma1)
    print(f'Gamma mean: {torch.mean(gamma1)}')
    print('beta')
    print(f'Beta mean: {torch.mean(beta)}')
    print(beta)
