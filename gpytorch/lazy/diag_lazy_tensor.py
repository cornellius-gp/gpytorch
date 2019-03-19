#!/usr/bin/env python3

import torch
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor


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

    @cached(name="cholesky")
    def _cholesky(self):
        return self.sqrt()

    def _cholesky_solve(self, rhs):
        return rhs / self._diag.pow(2)

    def _expand_batch(self, batch_shape):
        return self.__class__(self._diag.expand(*batch_shape, self._diag.size(-1)))

    def _get_indices(self, row_index, col_index, *batch_indices):
        res = self._diag[(*batch_indices, row_index)]
        # If row and col index don't agree, then we have off diagonal elements
        # Those should be zero'd out
        res = res * torch.eq(row_index, col_index).to(device=res.device, dtype=res.dtype)
        return res

    def _matmul(self, rhs):
        # to perform matrix multiplication with diagonal matrices we can just
        # multiply element-wise with the diagonal (using proper broadcasting)
        if rhs.ndimension() == 1:
            return self._diag * rhs
        return self._diag.unsqueeze(-1) * rhs

    def _mul_constant(self, constant):
        return self.__class__(self._diag * constant.unsqueeze(-1))

    def _mul_matrix(self, other):
        if isinstance(other, DiagLazyTensor):
            return self.__class__(self._diag * other._diag)
        else:
            return self.__class__(self._diag * other.diag())

    def _prod_batch(self, dim):
        return self.__class__(self._diag.prod(dim))

    def _quad_form_derivative(self, left_vecs, right_vecs):
        # TODO: Use proper batching for input vectors (prepand to shape rathern than append)
        res = left_vecs * right_vecs
        if res.ndimension() > self._diag.ndimension():
            res = res.sum(-1)
        return (res,)

    def _root_decomposition(self):
        return self.sqrt()

    def _root_inv_decomposition(self, initial_vectors=None):
        return DiagLazyTensor(self._diag.reciprocal()).sqrt()

    def _size(self):
        return self._diag.shape + self._diag.shape[-1:]

    def _sum_batch(self, dim):
        return self.__class__(self._diag.sum(dim))

    def _t_matmul(self, rhs):
        # Diagonal matrices always commute
        return self._matmul(rhs)

    def _transpose_nonbatch(self):
        return self

    def abs(self):
        return DiagLazyTensor(self._diag.abs())

    def add_diag(self, added_diag):
        return DiagLazyTensor(self._diag + added_diag.expand_as(self._diag))

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

    def sqrt(self):
        return DiagLazyTensor(self._diag.sqrt())

    def zero_mean_mvn_samples(self, num_samples):
        base_samples = torch.randn(num_samples, *self._diag.shape, dtype=self.dtype, device=self.device)
        return base_samples * self._diag.sqrt()
