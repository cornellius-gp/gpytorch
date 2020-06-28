#!/usr/bin/env python3

import torch

from ..utils import cached
from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .diag_lazy_tensor import DiagLazyTensor
from .kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from .non_lazy_tensor import lazify
from .root_lazy_tensor import RootLazyTensor


class KroneckerProductAddedDiagLazyTensor(AddedDiagLazyTensor):
    def __init__(self, *lazy_tensors, preconditioner_override=None):
        super(KroneckerProductAddedDiagLazyTensor, self).__init__(
            *lazy_tensors, preconditioner_override=preconditioner_override
        )
        if len(lazy_tensors) > 2:
            raise RuntimeError("An AddedDiagLazyTensor can only have two components")
        elif isinstance(lazy_tensors[0], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[0]
            self._lazy_tensor = lazy_tensors[1]
        elif isinstance(lazy_tensors[1], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[1]
            self._lazy_tensor = lazy_tensors[0]
        else:
            raise RuntimeError("One of the LazyTensors input to AddedDiagLazyTensor must be a DiagLazyTensor!")

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        if inv_quad_rhs is not None:
            inv_quad_term = self.inv_quad(inv_quad_rhs, reduce_inv_quad=reduce_inv_quad)
        else:
            inv_quad_term = None

        if logdet is not None:
            logdet_term = self.logdet()
        else:
            logdet_term = None

        return inv_quad_term, logdet_term

    def logdet(self):
        evals_plus_diag = self._kron_eigenvalues.diag() + self._diag_tensor.diag()
        return torch.log(evals_plus_diag).sum(dim=-1)

    @property
    @cached
    def _eig_cache(self):
        return [lt.evaluate().symeig(eigenvectors=True) for lt in self._lazy_tensor.lazy_tensors]

    @property
    @cached
    def _kron_eigenvalues(self):
        return KroneckerProductLazyTensor(*[DiagLazyTensor(eig_decomp[0]) for eig_decomp in self._eig_cache])

    def inv_quad(self, tensor, reduce_inv_quad=True):
        # TODO: check stability of numerics here

        q_matrix = KroneckerProductLazyTensor(*[lazify(eig_decomp[1]) for eig_decomp in self._eig_cache])
        inv_mat_sqrt = DiagLazyTensor(1.0 / (self._kron_eigenvalues.diag() + self._diag_tensor.diag()) ** 0.5)

        res = q_matrix.transpose(-2, -1).matmul(tensor)
        res2 = inv_mat_sqrt.matmul(res)

        if reduce_inv_quad:
            reduction_dims = (-1, -2)
        else:
            reduction_dims = -1

        final_res = res2.transpose(-2, -1).matmul(res2).sum(dim=reduction_dims)
        return final_res

    def sqrt_inv_matmul(self, rhs, lhs=None):
        q_matrix = KroneckerProductLazyTensor(*[lazify(eig_decomp[1]) for eig_decomp in self._eig_cache])
        inv_mat_sqrt = DiagLazyTensor(1.0 / (self._kron_eigenvalues.diag() + self._diag_tensor.diag()) ** 0.5)

        res = q_matrix.transpose(-2, -1).matmul(rhs)
        res2 = inv_mat_sqrt.matmul(res)

        if lhs is None:
            return q_matrix.matmul(res2)

        q_matrix_lhs = q_matrix.transpose(-2, -1).matmul(lhs.transpose(-2, -1)).transpose(-2, -1)
        sqrt_inv_matmul_res = q_matrix_lhs.matmul(res2)

        inv_matmul_res = self.inv_quad(lhs.transpose(-2, -1), reduce_inv_quad=False)
        return sqrt_inv_matmul_res, inv_matmul_res

    def _root_decomposition(self):
        q_matrix = KroneckerProductLazyTensor(*[lazify(eig_decomp[1]) for eig_decomp in self._eig_cache])
        eigs_sqrt = DiagLazyTensor((self._kron_eigenvalues.diag() + self._diag_tensor.diag()) ** 0.5)

        matrix_root = eigs_sqrt.matmul(q_matrix)

        return RootLazyTensor(matrix_root)

    # def _quad_form_derivative(self, left_vecs, right_vecs):
    #     res = left_vecs.matmul(right_vecs.transpose(-1, -2))
    #     return (res,)
