#!/usr/bin/env python3

import warnings
import torch


def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.cholesky(A, upper=upper, out=out)
        # TODO: Remove once fixed in pytorch (#16780)
        if A.dim() > 2 and A.is_cuda:
            if torch.isnan(L if out is None else out).any():
                raise RuntimeError
    except RuntimeError:
        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        idx = torch.arange(A.shape[-1], device=A.device)
        Aprime = A.clone()
        Aprime[..., idx, idx] += jitter
        try:
            L = torch.cholesky(Aprime, upper=upper, out=out)
            # TODO: Remove once fixed in pytorch (#16780)
            if A.dim() > 2 and A.is_cuda:
                if torch.isnan(L if out is None else out).any():
                    raise RuntimeError("singular")
        except RuntimeError as e:
            if "singular" in e.args[0]:
                raise RuntimeError("Adding jitter of {} to the diagonal did not make A p.d.".format(jitter))
        warnings.warn("A not p.d., added jitter of {} to the diagonal".format(jitter), RuntimeWarning)

    return L


def cholesky_solve(b, u, upper=False):
    if hasattr(torch, "cholesky_solve"):
        return torch.cholesky_solve(b, u, upper=False)
    else:
        return torch.potrs(b, u, upper=False)
