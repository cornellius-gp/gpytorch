#!/usr/bin/env python3

import torch

from ..utils import cached
from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .diag_lazy_tensor import DiagLazyTensor
from .kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from .non_lazy_tensor import lazify

# TODO: check the speed of the linear algebra - svd vs eig etc.


class KroneckerProductAddedDiagLazyTensor(AddedDiagLazyTensor):
    def __init__(self, *lazy_tensors):
        super(KroneckerProductAddedDiagLazyTensor, self).__init__(*lazy_tensors)
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
            inv_quad_term = self.inv_quad(inv_quad_rhs)
        else:
            inv_quad_term = None

        if logdet is not None:
            logdet_term = self.logdet()
        else:
            logdet_term = None

        return inv_quad_term, logdet_term

    def logdet(self):
        noisy_eigenvalues = self._kron_eigenvalues.diag() + self._diag_tensor.diag()
        return torch.log(noisy_eigenvalues).sum()

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
        inv_mat = DiagLazyTensor(1.0 / (self._kron_eigenvalues.diag() + self._diag_tensor.diag()))

        res = q_matrix.t().matmul(tensor)
        res = inv_mat.matmul(res)
        res = q_matrix.matmul(res)

        return tensor.t().matmul(res).squeeze()
