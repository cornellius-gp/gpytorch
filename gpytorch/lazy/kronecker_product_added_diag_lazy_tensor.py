#!/usr/bin/env python3

import torch

from .. import settings
from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .diag_lazy_tensor import DiagLazyTensor
from .kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor, lazify

# TODO: check the speed of the linear algebra - svd vs eig etc.


class _DiagKroneckerProdLazyTensor(KroneckerProductLazyTensor):
    def __init__(self, *lazy_tensors):
        try:
            lazy_tensors = tuple(lazify(lazy_tensor) for lazy_tensor in lazy_tensors)
        except TypeError:
            raise RuntimeError("DiagKronProductLazyTensor is intended to wrap lazy tensors.")
        for prev_lazy_tensor, curr_lazy_tensor in zip(lazy_tensors[:-1], lazy_tensors[1:]):
            if prev_lazy_tensor.batch_shape != curr_lazy_tensor.batch_shape:
                raise RuntimeError(
                    "DiagKronProductLazyTensor expects lazy tensors with the "
                    "same batch shapes. Got {}.".format([lv.batch_shape for lv in lazy_tensors])
                )
        super(_DiagKroneckerProdLazyTensor, self).__init__(*lazy_tensors)
        self.lazy_tensors = lazy_tensors

    def get_diag(self):
        sz1 = self.lazy_tensors[0].size(0)
        sz2 = self.lazy_tensors[1].size(0)

        d1 = self.lazy_tensors[0].diag()
        d2 = self.lazy_tensors[1].diag()

        out = d2.expand(sz1, sz2).t().mul(d1)

        return out.t().contiguous().view(sz1 * sz2)


class _KroneckerProductLazyLogDet(KroneckerProductLazyTensor):
    def __init__(self, *lazy_tensors, jitter=settings.tridiagonal_jitter.value()):
        super(_KroneckerProductLazyLogDet, self).__init__(*lazy_tensors)
        # on initialization take the eigenvectors & eigenvalues of all of the lazy tensors
        self.eig_cache = [torch.symeig(lt.evaluate(), eigenvectors=True) for lt in self.lazy_tensors]

    def inv_matmul(self, rhs, jitter=settings.tridiagonal_jitter.value()):
        Vinv = KroneckerProductLazyTensor(*[DiagLazyTensor(1 / (s[0].abs() + jitter)) for s in self.eig_cache])
        Q = KroneckerProductLazyTensor(*[NonLazyTensor(s[1]) for s in self.eig_cache])

        # first compute Q^T y
        res1 = Q.t().matmul(rhs)

        # now V^{-1} Q^T y
        res2 = Vinv.matmul(res1)
        res3 = Q.matmul(res2)

        return res3

    def logdet(self):
        lt_sizes = [lt.size(-1) for lt in self.lazy_tensors]

        # det(A \kron B) = det(A)^m det(B)^n where m,n are the sizes of A,B
        scaled_logdets = [m * s[0].sum() for m, s in zip(lt_sizes, self.eig_cache)]

        full_logdet = 0.0
        for logdet in scaled_logdets:
            full_logdet = logdet + full_logdet

        return full_logdet


class KroneckerProductAddedDiagLazyTensor(AddedDiagLazyTensor):
    def __init__(self, *lazy_tensors):
        super(KroneckerProductAddedDiagLazyTensor, self).__init__(*lazy_tensors)
        if len(lazy_tensors) > 2:
            raise RuntimeError("An AddedDiagLazyTensor can only have two components")
        if isinstance(lazy_tensors[0], DiagLazyTensor) and isinstance(lazy_tensors[1], DiagLazyTensor):
            raise RuntimeError("Trying to lazily add two DiagLazyTensors. " "Create a single DiagLazyTensor instead.")
        elif isinstance(lazy_tensors[0], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[0]
            self._lazy_tensor = lazy_tensors[1]
        elif isinstance(lazy_tensors[1], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[1]
            self._lazy_tensor = lazy_tensors[0]
        else:
            raise RuntimeError("One of the LazyTensors input to AddedDiagLazyTensor must be a DiagLazyTensor!")

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        # print("Calling the right inv quad logdet")
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
        sub_eigs = []
        for lt in self._lazy_tensor.lazy_tensors:
            sub_eigs.append(lt.evaluate().eig()[0][:, 0].unsqueeze(-1))

        eigs = sub_eigs[0].matmul(sub_eigs[1].t())
        return torch.log(eigs + noise).sum()

    def _kron_svd(self):
        return [lt.evaluate().svd() for lt in self._lazy_tensor.lazy_tensors]

    def inv_quad(self, rhs):
        # TODO: check stability of numerics here

        svd_list = self._kron_svd()
        noise = self._diag_tensor[0, 0]
        V = _DiagKroneckerProdLazyTensor(DiagLazyTensor(svd_list[0].S), DiagLazyTensor(svd_list[1].S))
        Q = KroneckerProductLazyTensor(lazify(svd_list[0].U), lazify(svd_list[1].U))
        for sub_ind in range(2, len(svd_list)):
            V = KroneckerProductLazyTensor(V, DiagLazyTensor(svd_list[sub_ind].S))
            Q = KroneckerProductLazyTensor(Q, LazyTensor(svd_list[sub_ind].S))

        # TODO: this could be a memory hog.
        inv_mat = DiagLazyTensor(V.get_diag() + noise)

        res = Q.t().matmul(rhs)
        res = inv_mat.inverse().matmul(res)
        res = Q.matmul(res)

        return rhs.t().matmul(res).squeeze()
