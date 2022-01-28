#!/usr/bin/env python3

import typing  # noqa F401

from ..utils.memoize import cached
from .root_lazy_tensor import RootLazyTensor
from .triangular_lazy_tensor import TriangularLazyTensor


class InverseCholLazyTensor(CholLazyTensor):
    def __init__(self, chol: TriangularLazyTensor, upper: bool = False):
        super().__init__(chol)
        self.upper = upper

    def _solve(self, rhs, preconditioner, num_tridiag=0):
        if num_tridiag:
            return super()._solve(rhs, preconditioner, num_tridiag=num_tridiag)
        return self.root._cholesky_solve(rhs, upper=self.upper)

    def _matmul(self, rhs):
        return self.root._matmul(self.root._t_matmul(rhs))

    def inv_matmul(self, right_tensor, left_tensor=None):
        is_vector = right_tensor.ndim == 1
        if is_vector:
            right_tensor = right_tensor.unsqueeze(-1)
        res = self.root._cholesky_solve(right_tensor, upper=self.upper)
        if is_vector:
            res = res.squeeze(-1)
        if left_tensor is not None:
            res = left_tensor @ res
        return res
