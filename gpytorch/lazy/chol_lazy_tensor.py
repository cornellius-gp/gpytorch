#!/usr/bin/env python3

import torch

from .lazy_tensor import delazify
from .batch_repeat_lazy_tensor import BatchRepeatLazyTensor
from .root_lazy_tensor import RootLazyTensor
from .. import settings


class CholLazyTensor(RootLazyTensor):
    def __init__(self, chol):
        # Check that we have a lower triangular matrix
        if settings.debug.on():
            chol = delazify(chol) if not isinstance(chol, BatchRepeatLazyTensor) else delazify(chol.base_lazy_tensor)
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

    def _cholesky(self):
        return self.root

    def _solve(self, rhs, preconditioner, num_tridiag=0):
        if num_tridiag:
            return super()._solve(rhs, preconditioner, num_tridiag=num_tridiag)
        else:
            return self.root._cholesky_solve(rhs)

    def inv_matmul(self, right_tensor, left_tensor=None):
        with settings.fast_computations(solves=False):
            return super().inv_matmul(right_tensor, left_tensor=left_tensor)

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        if not self.is_square:
            raise RuntimeError(
                "inv_quad_logdet only operates on (batches of) square (positive semi-definite) LazyTensors. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if inv_quad_rhs is not None:
            if self.dim() == 2 and inv_quad_rhs.dim() == 1:
                if self.shape[-1] != inv_quad_rhs.numel():
                    raise RuntimeError(
                        "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                            self.shape, inv_quad_rhs.shape
                        )
                    )
            elif self.dim() != inv_quad_rhs.dim():
                raise RuntimeError(
                    "LazyTensor (size={}) and right-hand-side Tensor (size={}) should have the same number "
                    "of dimensions.".format(self.shape, inv_quad_rhs.shape)
                )
            elif self.shape[-1] != inv_quad_rhs.shape[-2]:
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, inv_quad_rhs.shape
                    )
                )

        inv_quad_term = None
        logdet_term = None

        if inv_quad_rhs is not None:
            inv_quad_term = self.inv_quad(inv_quad_rhs, reduce_inv_quad=reduce_inv_quad)

        if logdet:
            logdet_term = self._chol_diag.pow(2).log().sum(-1)

        return inv_quad_term, logdet_term
