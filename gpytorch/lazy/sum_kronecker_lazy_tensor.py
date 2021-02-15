#!/usr/bin/env python3

from .. import settings
from .kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from .sum_lazy_tensor import SumLazyTensor


class SumKroneckerLazyTensor(SumLazyTensor):
    r"""
    Returns the sum of two Kronecker product lazy tensors. Solves and log-determinants
    are computed using the eigen-decomposition of the right lazy tensor; that is,
    (A \kron B + C \kron D) = (C^{1/2} \kron D^{1/2})
        (C^{-1/2}AC^{-1/2} \kron D^{-1/2} B D^{-1/2} + I_|C| \kron I_|D|)(C^{1/2} \kron D^{1/2})^{T}
    where .^{1/2} = Q_.\Lambda_.^{1/2} (e.g. an eigen-decomposition.)

    This formulation admits efficient solves and log determinants.

    The original reference is [https://papers.nips.cc/paper/2013/file/59c33016884a62116be975a9bb8257e3-Paper.pdf].

    Args:
        :`lazy_tensors`: List of two Kronecker lazy tensors
    """

    @property
    def _sum_formulation(self):
        # where M = (C^{-1/2}AC^{-1/2} \kron D^{-1/2} B D^{-1/2} + I_|C| \kron I_|D|)
        lt1 = self.lazy_tensors[0]
        lt2 = self.lazy_tensors[1]

        lt2_inv_roots = [lt.root_inv_decomposition().root for lt in lt2.lazy_tensors]

        lt2_inv_root_mm_lt2 = [
            rm.transpose(-1, -2).matmul(lt).matmul(rm) for rm, lt in zip(lt2_inv_roots, lt1.lazy_tensors)
        ]
        inv_root_times_lt1 = KroneckerProductLazyTensor(*lt2_inv_root_mm_lt2).add_jitter(1.0)
        return inv_root_times_lt1

    def _solve(self, rhs, preconditioner=None, num_tridiag=0):
        if self.shape[-1] <= settings.max_cholesky_size.value():
            return super()._solve(rhs=rhs, preconditioner=preconditioner, num_tridiag=num_tridiag)

        inner_mat = self._sum_formulation
        # root decomposition may not be trustworthy if it uses a different method than
        # root_inv_decomposition. so ensure that we call this locally
        lt2_inv_roots = [lt.root_inv_decomposition().root for lt in self.lazy_tensors[1].lazy_tensors]
        lt2_inv_root = KroneckerProductLazyTensor(*lt2_inv_roots)

        # now we compute L^{-1} M L^{-T} z
        # where M = (C^{-1/2}AC^{-1/2} \kron D^{-1/2} B D^{-1/2} + I_|C| \kron I_|D|)
        res = lt2_inv_root.transpose(-1, -2).matmul(rhs)
        res = inner_mat.inv_matmul(res)
        res = lt2_inv_root.matmul(res)

        return res

    def _logdet(self):
        inner_mat = self._sum_formulation
        lt2_logdet = self.lazy_tensors[1].logdet()
        return inner_mat._logdet() + lt2_logdet

    def _root_decomposition(self):
        inner_mat = self._sum_formulation
        lt2_root = KroneckerProductLazyTensor(
            *[lt.root_decomposition().root for lt in self.lazy_tensors[1].lazy_tensors]
        )
        inner_mat_root = inner_mat.root_decomposition().root
        root = lt2_root.matmul(inner_mat_root)
        return root

    def _root_inv_decomposition(self, initial_vectors=None):
        inner_mat = self._sum_formulation
        lt2_root_inv = self.lazy_tensors[1].root_inv_decomposition().root
        inner_mat_root_inv = inner_mat.root_inv_decomposition().root
        inv_root = lt2_root_inv.matmul(inner_mat_root_inv)
        return inv_root
