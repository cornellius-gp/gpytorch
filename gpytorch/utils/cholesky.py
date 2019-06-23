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

        return L
    except RuntimeError as e:
        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        jitter_ori = jitter
        for i in range(3):
            jitter = jitter_ori * (10 ** i)
            Aprime = A.clone()
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter)
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                # TODO: Remove once fixed in pytorch (#16780)
                if A.dim() > 2 and A.is_cuda:
                    if torch.isnan(L if out is None else out).any():
                        raise RuntimeError("singular")
                warnings.warn(f"A not p.d., added jitter of {jitter} to the diagonal", RuntimeWarning)
                return L
            except RuntimeError:
                continue

        raise e
