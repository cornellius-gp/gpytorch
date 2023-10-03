#!/usr/bin/env python3
import torch

from .. import settings
from .cholesky import psd_safe_cholesky


def alternating_projection(
    train_x, covar_module, noise, rhs,
    batch, maxiter=None, tolerance=None,
    tracker=None,
):
    """
    This implementation uses the default partition and makes the following easy:
    1. batch Cholesky decomposition
    2. fast Gauss-Southwell rule implementation
    3. support multiple right hand sides
    """
    if rhs.dim() == 1:
        rhs = rhs.unsqueeze(-1)

    if maxiter is None:
        maxiter = settings.max_cg_iterations.value()

    if tolerance is None:
        if settings._use_eval_tolerance.on():
            tolerance = settings.eval_cg_tolerance.value()
        else:
            tolerance = settings.cg_tolerance.value()

    n, d = train_x.size()
    _, m = rhs.size()

    num_batch = n // batch
    remainder = n % batch
    assert remainder > 0

    normalized_rhs = rhs / rhs.norm(dim=-2, keepdim=True)
    r = normalized_rhs.detach().clone()

    weights = torch.zeros_like(rhs)

    # sub_mats = sub_mats.add_jitter(noise).evaluate() # memory overflow
    sub_mats = covar_module(train_x[:-remainder].view(num_batch, batch, d)).evaluate()
    sub_mats += noise * torch.eye(batch, dtype=rhs.dtype, device=rhs.device)
    batch_chol = psd_safe_cholesky(sub_mats)

    last_mat = covar_module(train_x[-remainder:]).add_jitter(noise).evaluate()
    last_chol = psd_safe_cholesky(last_mat)

    for i in range(maxiter):
        for j in range(num_batch):
            k = r[:-remainder].T.view(-1, num_batch, batch).norm(dim=-1).mean(dim=0).argmax()
            indices = slice(k * batch, (k + 1) * batch)

            updates = torch.cholesky_solve(r[indices], batch_chol[k], upper=False)
            weights[indices] += updates

            r -= covar_module(train_x, train_x[indices]) @ updates
            r[indices] -= noise * updates

        """
        Update the last block...
        """
        updates = torch.cholesky_solve(r[-remainder:], last_chol, upper=False)
        weights[-remainder:] += updates

        r -= covar_module(train_x, train_x[-remainder:]) @ updates
        r[-remainder:] -= noise * updates

        # record residual
        avg_residual_norm = torch.linalg.norm(r, dim=-2).mean().item()

        if tracker is not None:
            tracker.log({'residual': avg_residual_norm})

        if settings.verbose.on():
            print("iter {:4d}, avg residual {:f}".format(i, avg_residual_norm))

        if settings.record_residual.on():
            settings.record_residual.lst_residual_norm.append(avg_residual_norm)

        if i >= 10 and avg_residual_norm < tolerance:
            break

    return weights * rhs.norm(dim=-2, keepdim=True)
