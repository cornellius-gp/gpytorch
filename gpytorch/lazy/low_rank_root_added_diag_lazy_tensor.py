#!/usr/bin/env python3

import torch

from . import delazify
from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .diag_lazy_tensor import DiagLazyTensor
from .low_rank_root_lazy_tensor import LowRankRootLazyTensor
from ..utils.cholesky import psd_safe_cholesky
from ..utils.memoize import cached


class LowRankRootAddedDiagLazyTensor(AddedDiagLazyTensor):
    def __init__(self, *lazy_tensors, preconditioner_override=None):
        if len(lazy_tensors) > 2:
            raise RuntimeError("An AddedDiagLazyTensor can only have two components")

        if isinstance(lazy_tensors[0], DiagLazyTensor) and not isinstance(lazy_tensors[1], LowRankRootLazyTensor):
            raise RuntimeError("A LowRankRootAddedDiagLazyTensor can only be created with a LowRankLazyTensor base!")
        elif isinstance(lazy_tensors[1], DiagLazyTensor) and not isinstance(lazy_tensors[0], LowRankRootLazyTensor):
            raise RuntimeError("A LowRankRootAddedDiagLazyTensor can only be created with a LowRankLazyTensor base!")

        super().__init__(*lazy_tensors, preconditioner_override=preconditioner_override)

    @cached(name="chol_cap_mat")
    def chol_cap_mat(self):
        A_inv = self._diag_tensor.inverse()  # This is fine since it's a DiagLazyTensor
        U = self._lazy_tensor.root
        V = self._lazy_tensor.root.transpose(-2, -1)
        C = DiagLazyTensor(torch.ones(*V.batch_shape, V.shape[-2], V.shape[-2], device=V.device, dtype=V.dtype))

        cap_mat = delazify(C + V.matmul(A_inv.matmul(U)))
        chol_cap_mat = psd_safe_cholesky(cap_mat)

        return chol_cap_mat

    def _solve(self, rhs, preconditioner=None, num_tridiag=0):
        A_inv = self._diag_tensor.inverse()  # This is fine since it's a DiagLazyTensor
        U = self._lazy_tensor.root
        V = self._lazy_tensor.root.transpose(-2, -1)
        chol_cap_mat = self.chol_cap_mat

        res = V.matmul(A_inv.matmul(rhs))
        res = torch.cholesky_solve(res, chol_cap_mat)
        res = A_inv.matmul(U.matmul(res))

        solve = A_inv.matmul(rhs) - res

        return solve

    def _logdet(self):
        chol_cap_mat = self.chol_cap_mat
        logdet_cap_mat = 2 * chol_cap_mat.diag().log().sum(-1)
        logdet_A = self._diag_tensor.logdet()
        logdet_term = logdet_cap_mat + logdet_A

        return logdet_term

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
            elif self.batch_shape != inv_quad_rhs.shape[:-2] or self.shape[-1] != inv_quad_rhs.shape[-2]:
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, inv_quad_rhs.shape
                    )
                )

        inv_quad_term, logdet_term = None, None

        if inv_quad_rhs is not None:
            self_inv_rhs = self._solve(inv_quad_rhs)
            inv_quad_term = inv_quad_rhs.transpose(-2, -1).matmul(self_inv_rhs)

        if logdet:
            logdet_term = self._logdet

        return inv_quad_term, logdet_term
