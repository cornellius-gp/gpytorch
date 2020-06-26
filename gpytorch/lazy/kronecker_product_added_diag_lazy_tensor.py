#!/usr/bin/env python3

import torch

# from .. import settings
# from ..utils import cached
from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .diag_lazy_tensor import DiagLazyTensor
from .kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from .non_lazy_tensor import lazify

# TODO: check the speed of the linear algebra - svd vs eig etc.


# class _KroneckerProductLazyLogDet(KroneckerProductLazyTensor):
#     def __init__(self, *lazy_tensors, jitter=settings.tridiagonal_jitter.value()):
#         super(_KroneckerProductLazyLogDet, self).__init__(*lazy_tensors)
#         # on initialization take the eigenvectors & eigenvalues of all of the lazy tensors
#         self.eig_cache = [torch.symeig(lt.evaluate(), eigenvectors=True) for lt in self.lazy_tensors]

#     def inv_matmul(self, rhs, jitter=settings.tridiagonal_jitter.value()):
#         Vinv = KroneckerProductLazyTensor(*[DiagLazyTensor(1 / (s[0].abs() + jitter)) for s in self.eig_cache])
#         Q = KroneckerProductLazyTensor(*[NonLazyTensor(s[1]) for s in self.eig_cache])

#         # first compute Q^T y
#         res1 = Q.t().matmul(rhs)

#         # now V^{-1} Q^T y
#         res2 = Vinv.matmul(res1)
#         res3 = Q.matmul(res2)

#         return res3

#     def logdet(self):
#         lt_sizes = [lt.size(-1) for lt in self.lazy_tensors]

#         # det(A \kron B) = det(A)^m det(B)^n where m,n are the sizes of A,B
#         scaled_logdets = [m * s[0].sum() for m, s in zip(lt_sizes, self.eig_cache)]

#         full_logdet = 0.0
#         for logdet in scaled_logdets:
#             full_logdet = logdet + full_logdet

#         return full_logdet


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
        noise = self._diag_tensor[0, 0]
        sub_eigs = [DiagLazyTensor(svd_decomp.S) for svd_decomp in self._kron_svd]
        sub_eigs_kronecker = KroneckerProductLazyTensor(*sub_eigs).diag()
        return torch.log(sub_eigs_kronecker + noise).sum()

    @property
    def _kron_svd(self):
        return [lt.evaluate().svd() for lt in self._lazy_tensor.lazy_tensors]

    def inv_quad(self, tensor, reduce_inv_quad=True):
        # TODO: check stability of numerics here
        svd_list = self._kron_svd
        v_matrix = KroneckerProductLazyTensor(*[DiagLazyTensor(svd_decomp.S) for svd_decomp in svd_list])
        q_matrix = KroneckerProductLazyTensor(*[lazify(svd_decomp.U) for svd_decomp in svd_list])

        # TODO: this could be a memory hog.
        inv_mat = DiagLazyTensor(1.0 / (v_matrix.diag() + self._diag_tensor.diag()))

        res = q_matrix.t().matmul(tensor)
        res = inv_mat.matmul(res)
        res = q_matrix.matmul(res)

        return tensor.t().matmul(res).squeeze()
