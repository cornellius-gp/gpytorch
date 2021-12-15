#!/usr/bin/env python3

import warnings

import torch

from .. import settings
from .errors import NanError, NotPSDError
from .warnings import NumericalWarning


def _psd_safe_cholesky(A, out=None, jitter=None, max_tries=None):
    # Maybe log
    if settings.verbose_linalg.on():
        settings.verbose_linalg.logger.debug(f"Running Cholesky on a matrix of size {A.shape}.")

    if out is not None:
        out = (out, torch.empty(A.shape[:-2], dtype=torch.int32, device=out.device))

    L, info = torch.linalg.cholesky_ex(A, out=out)
    if not torch.any(info):
        return L

    isnan = torch.isnan(A)
    if isnan.any():
        raise NanError(f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN.")

    if jitter is None:
        jitter = settings.cholesky_jitter.value(A.dtype)
    if max_tries is None:
        max_tries = settings.cholesky_max_tries.value()
    Aprime = A.clone()
    jitter_prev = 0
    for i in range(max_tries):
        jitter_new = jitter * (10 ** i)
        # add jitter only where needed
        diag_add = ((info > 0) * (jitter_new - jitter_prev)).unsqueeze(-1).expand(*Aprime.shape[:-1])
        Aprime.diagonal(dim1=-1, dim2=-2).add_(diag_add)
        jitter_prev = jitter_new
        warnings.warn(
            f"A not p.d., added jitter of {jitter_new:.1e} to the diagonal",
            NumericalWarning,
        )
        L, info = torch.linalg.cholesky_ex(Aprime, out=out)
        if not torch.any(info):
            return L
    raise NotPSDError(f"Matrix not positive definite after repeatedly adding jitter up to {jitter_new:.1e}.")


def psd_safe_cholesky(A, upper=False, out=None, jitter=None, max_tries=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted,
            uses settings.cholesky_jitter.value()
        :attr:`max_tries` (int, optional):
            Number of attempts (with successively increasing jitter) to make before raising an error.
    """
    L = _psd_safe_cholesky(A, out=out, jitter=jitter, max_tries=max_tries)
    if upper:
        if out is not None:
            out = out.transpose_(-1, -2)
        else:
            L = L.transpose(-1, -2)
    return L
