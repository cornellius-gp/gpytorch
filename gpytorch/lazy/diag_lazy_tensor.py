#!/usr/bin/env python3

import torch
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify
from .root_lazy_tensor import RootLazyTensor


class DiagLazyTensor(LazyTensor):
    def __init__(self, diag):
        """
        Diagonal lazy tensor. Supports arbitrary batch sizes.

        Args:
            :attr:`diag` (Tensor):
                A `b1 x ... x bk x n` Tensor, representing a `b1 x ... x bk`-sized batch
                of `n x n` diagonal matrices
        """
        super(DiagLazyTensor, self).__init__(diag)
        self._diag = diag

    def __add__(self, other):
        if isinstance(other, DiagLazyTensor):
            return self.add_diag(other._diag)
        from .added_diag_lazy_tensor import AddedDiagLazyTensor

        return AddedDiagLazyTensor(other, self)

    def _expand_batch(self, batch_shape):
        return self.__class__(self._diag.expand(*batch_shape, self._diag.size(-1)))

    def _matmul(self, rhs):
        # to perform matrix multiplication with diagonal matrices we can just
        # multiply element-wise with the diagonal (using proper broadcasting)
        if rhs.ndimension() == 1:
            return self._diag * rhs
        return self._diag.unsqueeze(-1) * rhs

    def _quad_form_derivative(self, left_vecs, right_vecs):
        # TODO: Use proper batching for input vectors (prepand to shape rathern than append)
        res = left_vecs * right_vecs
        if res.ndimension() > self._diag.ndimension():
            res = res.sum(-1)
        return (res,)

    def _size(self):
        return self._diag.shape + self._diag.shape[-1:]

    def _sum_batch(self, dim):
        return DiagLazyTensor(self._diag.sum(dim))

    def _t_matmul(self, rhs):
        # Diagonal matrices always commute
        return self._matmul(rhs)

    def _transpose_nonbatch(self):
        return self

    def abs(self):
        return DiagLazyTensor(self._diag.abs())

    def add_diag(self, added_diag):
        return DiagLazyTensor(self._diag + added_diag.expand_as(self._diag))

    def __mul__(self, other):
        other = lazify(other)
        if isinstance(other, DiagLazyTensor):
            return DiagLazyTensor(self._diag * other._diag)
        else:
            other_diag = other.diag()
            new_diag = self._diag * other_diag
            corrected_diag = new_diag - other_diag

            return other.add_diag(corrected_diag)

    def diag(self):
        return self._diag

    @cached
    def evaluate(self):
        return self._diag.unsqueeze(-1) * torch.eye(self._diag.shape[-1], dtype=self.dtype, device=self.device)

    def exp(self):
        return DiagLazyTensor(self._diag.exp())

    def inverse(self):
        return DiagLazyTensor(self._diag.reciprocal())

    def inv_matmul(self, right_tensor, left_tensor=None):
        res = self.inverse()._matmul(right_tensor)
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):

        # TODO: Use proper batching for inv_quad_rhs (prepand to shape rathern than append)
        if inv_quad_rhs is None:
            rhs_batch_shape = torch.Size()
        else:
            rhs_batch_shape = inv_quad_rhs.shape[1 + self.batch_dim :]

        if inv_quad_rhs is None:
            inv_quad_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            diag = self._diag
            for _ in rhs_batch_shape:
                diag = diag.unsqueeze(-1)
            inv_quad_term = inv_quad_rhs.div(diag).mul(inv_quad_rhs).sum(-(1 + len(rhs_batch_shape)))
            if reduce_inv_quad:
                inv_quad_term = inv_quad_term.sum(-1)

        if not logdet:
            logdet_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            logdet_term = self._diag.log().sum(-1)

        return inv_quad_term, logdet_term

    def log(self):
        return DiagLazyTensor(self._diag.log())

    def matmul(self, other):
        # this is trivial if we multiply two DiagLazyTensors
        if isinstance(other, DiagLazyTensor):
            return DiagLazyTensor(self._diag * other._diag)
        return super(DiagLazyTensor, self).matmul(other)

    def root_decomposition(self):
        return RootLazyTensor(self.sqrt())

    def root_inv_decomposition(self):
        return RootLazyTensor(DiagLazyTensor(self._diag.reciprocal()).sqrt())

    def sqrt(self):
        return DiagLazyTensor(self._diag.sqrt())

    def zero_mean_mvn_samples(self, sample_shape=torch.Size()):
        base_samples = torch.randn(sample_shape + self._diag.shape, dtype=self.dtype, device=self.device)
        return base_samples * self._diag.sqrt()
