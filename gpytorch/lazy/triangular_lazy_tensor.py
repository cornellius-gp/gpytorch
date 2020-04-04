#!/usr/bin/env python3

import torch

from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.memoize import cached
from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor


class TriangularLazyTensor(LazyTensor):
    def __init__(self, tensor, upper=False):
        """
        Triangular lazy tensor. Supports arbitrary batch sizes.

        Args:
            :attr:`tensor` (Tensor or LazyTensor):
                A `b1 x ... x bk x n x n` Tensor, representing a `b1 x ... x bk`-sized batch
                of `n x n` triangular matrices.
            :attr:`upper` (bool):
                If True, the tensor is upper-triangular, otherwise lower-triangular.
        """
        super().__init__(tensor)
        if torch.is_tensor(tensor):
            tensor = NonLazyTensor(tensor)
        self._upper = upper
        self._tensor = tensor

    def __add__(self, other):
        if isinstance(other, TriangularLazyTensor) and not self._upper ^ other.upper:
            return self.__class__(self._tensor + other._tensor, upper=self._upper)
        return self._tensor + other

    @cached(name="cholesky")
    def _cholesky(self):
        return self.__class__(self._tensor._cholesky(), upper=self._upper)

    def _mul_constant(self, constant):
        return self.__class__(self._tensor * constant.unsqueeze(-1), upper=self._upper)

    def _root_decomposition(self):
        return self._tensor._root_decomposition()

    def _root_inv_decomposition(self, initial_vectors=None):
        self._tensor._root_inv_decomposition(initial_vectors=initial_vectors)

    def _size(self):
        return self._tensor.shape

    def _sum_batch(self, dim):
        return self.__class__(self._tensor._sum_batch(dim), upper=self._upper)

    def abs(self):
        return self.__class__(self._tensor.abs(), upper=self._upper)

    def add_diag(self, added_diag):
        shape = _mul_broadcast_shape(self._diag.shape, added_diag.shape)
        return self.__class__(
            AddedDiagLazyTensor(self._tensor.expand(shape), added_diag.expand(shape)), upper=self._upper
        )

    def diag(self):
        return self._tensor.diag()

    @cached
    def evaluate(self):
        return self._tensor.evaluate()

    def exp(self):
        return self.__class__(self._tensor.exp(), upper=self._upper)

    def inverse(self):
        eye = torch.eye(self._tensor.shape[-2:], device=self._tensor.device, dtype=self._tensor.dtype)
        inv = self.inv_matmul(eye)
        return self.__class__(inv, upper=self._upper)

    def inv_matmul(self, right_tensor, left_tensor=None):
        tsr = self.evaluate()
        res = torch.triangular_solve(right_tensor, tsr, upper=self._upper).solution
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        # triangular, inv_matmul is cheap
        inv_quad_term = inv_quad_rhs.transpose(-1, -2) @ self.inv_matmul(inv_quad_rhs)
        logdet_term = self.diag().prod(-1) if logdet else None
        if inv_quad_term.numel() and reduce_inv_quad:
            inv_quad_term = inv_quad_term.sum(-1)
        return inv_quad_term, logdet_term

    def _matmul(self, rhs):
        return self._tensor.matmul(rhs)

    def _solve(self, rhs, preconditioner, num_tridiag=0):
        # already triangular, can just call inv_matmul for the solve
        return self.inv_matmul(rhs)

    def _transpose_nonbatch(self):
        return self.__class__(self._tensor._transpose_nonbatch(), upper=not self._upper)
