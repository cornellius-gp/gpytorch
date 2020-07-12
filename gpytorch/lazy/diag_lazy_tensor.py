#!/usr/bin/env python3

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.memoize import cached
from .non_lazy_tensor import NonLazyTensor
from .triangular_lazy_tensor import TriangularLazyTensor


class DiagLazyTensor(TriangularLazyTensor):
    def __init__(self, diag):
        """
        Diagonal lazy tensor. Supports arbitrary batch sizes.

        Args:
            :attr:`diag` (Tensor):
                A `b1 x ... x bk x n` Tensor, representing a `b1 x ... x bk`-sized batch
                of `n x n` diagonal matrices
        """
        super(TriangularLazyTensor, self).__init__(diag)
        self._diag = diag

    def __add__(self, other):
        if isinstance(other, DiagLazyTensor):
            return self.add_diag(other._diag)
        from .added_diag_lazy_tensor import AddedDiagLazyTensor

        return AddedDiagLazyTensor(other, self)

    @cached(name="cholesky", ignore_args=True)
    def _cholesky(self, upper=False):
        return self.sqrt()

    def _cholesky_solve(self, rhs):
        return rhs / self._diag.unsqueeze(-1).pow(2)

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
        # special case if we have a NonLazyTensor
        if isinstance(rhs, NonLazyTensor):
            return NonLazyTensor(self._diag.unsqueeze(-1) * rhs.tensor)
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
        if not self._diag.requires_grad:
            return (None,)

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
        shape = _mul_broadcast_shape(self._diag.shape, added_diag.shape)
        return DiagLazyTensor(self._diag.expand(shape) + added_diag.expand(shape))

    def diag(self):
        return self._diag

    @cached
    def evaluate(self):
        if self._diag.dim() == 0:
            return self._diag
        return torch.diag_embed(self._diag)

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
        from .triangular_lazy_tensor import TriangularLazyTensor

        # this is trivial if we multiply two DiagLazyTensors
        if isinstance(other, DiagLazyTensor):
            return DiagLazyTensor(self._diag * other._diag)
        # special case if we have a NonLazyTensor
        if isinstance(other, NonLazyTensor):
            return NonLazyTensor(self._diag.unsqueeze(-1) * other.tensor)
        # and if we have a triangular one
        if isinstance(other, TriangularLazyTensor):
            return TriangularLazyTensor(self._diag.unsqueeze(-1) * other._tensor, upper=other.upper)
        return super().matmul(other)

    def sqrt(self):
        return DiagLazyTensor(self._diag.sqrt())

    def sqrt_inv_matmul(self, rhs, lhs=None):
        if lhs is None:
            return DiagLazyTensor(1.0 / (self._diag.sqrt())).matmul(rhs)
        else:
            matrix_inv_root = DiagLazyTensor(1.0 / (self._diag.sqrt()))
            sqrt_inv_matmul = lhs @ DiagLazyTensor(1.0 / (self._diag.sqrt())).matmul(rhs)
            inv_quad = (matrix_inv_root @ lhs.transpose(-2, -1)).transpose(-2, -1).pow(2).sum(dim=-1)

            return sqrt_inv_matmul, inv_quad

    def zero_mean_mvn_samples(self, num_samples):
        base_samples = torch.randn(num_samples, *self._diag.shape, dtype=self.dtype, device=self.device)
        return base_samples * self._diag.sqrt()

    @cached(name="svd")
    def _svd(self) -> Tuple[LazyTensor, Tensor, LazyTensor]:
        evals, evecs = self.symeig(eigenvectors=True)
        S = torch.abs(evals)
        U = evecs
        V = evecs * torch.sign(evals).unsqueeze(-1)
        return U, S, V

    @cached(name="symeig")
    def _symeig(self, eigenvectors: bool = False) -> Tuple[Tensor, Optional[LazyTensor]]:
        evals = self._diag
        if eigenvectors:
            evecs = DiagLazyTensor(torch.ones_like(evals))
        else:
            evecs = None
        return evals, evecs


class ConstantDiagLazyTensor(DiagLazyTensor):
    def __init__(self, diag_values, diag_shape):
        """
        Diagonal lazy tensor with constant entries. Supports arbitrary batch sizes.
        Used e.g. for adding jitter to matrices.

        Args:
            :attr:`n` (int):
                The (non-batch) dimension of the (square) matrix
            :attr:`diag_values` (Tensor):
                A `b1 x ... x bk x 1` Tensor, representing a `b1 x ... x bk`-sized batch
                of `n x n` diagonal matrices
        """
        super(TriangularLazyTensor, self).__init__(diag_values, diag_shape=diag_shape)
        self.diag_shape = diag_shape
        self._diag = diag_values.expand(*diag_values.shape[:-1], diag_shape)

    def _expand_batch(self, batch_shape):
        return self.__class__(self._diag.expand(*batch_shape, self._diag.size(-1)), diag_shape=self.diag_shape)

    def _sum_batch(self, dim):
        return self.__class__(self._diag.sum(dim), diag_shape=self.diag_shape)
