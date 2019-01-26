#!/usr/bin/env python3

import torch

from .lazy_tensor import LazyTensor
from .root_lazy_tensor import RootLazyTensor
from .. import settings


class CholLazyTensor(RootLazyTensor):
    def __init__(self, chol):
        if isinstance(chol, LazyTensor):  # Probably is an instance of NonLazyTensor
            chol = chol.evaluate()

        # Check that we have a lower triangular matrix
        if settings.debug.on():
            mask = torch.ones(chol.shape[-2:], dtype=chol.dtype, device=chol.device).triu_(1)
            if torch.max(chol.mul(mask)).item() > 1e-3 and torch.equal(chol, chol):
                raise RuntimeError("CholLazyVaraiable should take a lower-triangular matrix in the constructor.")

        # Run super constructor
        super(CholLazyTensor, self).__init__(chol)

    @property
    def _chol(self):
        if not hasattr(self, "_chol_memo"):
            self._chol_memo = self.root.evaluate()
        return self._chol_memo

    @property
    def _chol_diag(self):
        if not hasattr(self, "_chol_diag_memo"):
            self._chol_diag_memo = self._chol.diagonal(dim1=-2, dim2=-1).clone()
        return self._chol_diag_memo

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        inv_quad_term = None
        logdet_term = None

        if inv_quad_rhs is not None:
            inv_quad_term, _ = super(CholLazyTensor, self).inv_quad_logdet(
                inv_quad_rhs, logdet=False, reduce_inv_quad=reduce_inv_quad
            )

        if logdet:
            logdet_term = self._chol_diag.pow(2).log().sum(-1)

        return inv_quad_term, logdet_term
