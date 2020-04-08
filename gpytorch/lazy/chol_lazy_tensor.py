#!/usr/bin/env python3

import typing  # noqa F401

from ..utils.memoize import cached
from .root_lazy_tensor import RootLazyTensor
from .triangular_lazy_tensor import TriangularLazyTensor


class CholLazyTensor(RootLazyTensor):
    def __init__(self, chol: TriangularLazyTensor, upper: bool = False):
        super().__init__(chol)
        self.upper = upper

    @property
    def _chol_diag(self):
        return self.root.diag()

    def _cholesky(self, upper=False):
        if upper == self.upper:
            return self.root
        else:
            return self.root.transpose(-1, -2)

    def _solve(self, rhs, preconditioner, num_tridiag=0):
        if num_tridiag:
            return super()._solve(rhs, preconditioner, num_tridiag=num_tridiag)
        return self.root._cholesky_solve(rhs, upper=self.upper)

    @cached
    def evaluate(self):
        root = self.root
        if self.upper:
            res = root.transpose(-1, -2) @ root
        else:
            res = root @ root.transpose(-1, -2)
        return res.evaluate()

    @cached
    def inverse(self):
        Linv = self.root.inverse()
        return CholLazyTensor(Linv, upper=not self.upper)

    def inv_matmul(self, right_tensor, left_tensor=None):
        is_vector = right_tensor.ndim == 1
        if is_vector:
            right_tensor = right_tensor.unsqueeze(-1)
        res = self.root._cholesky_solve(right_tensor, upper=self.upper)
        if left_tensor is not None:
            res = left_tensor @ res
        if is_vector:
            res = res.squeeze(-1)
        return res

    def inv_quad(self, tensor, reduce_inv_quad=True):
        if self.root.upper:
            R = self.root._transpose_nonbatch().inv_matmul(tensor)
        else:
            R = self.root.inv_matmul(tensor)
        inv_quad_term = R.transpose(-1, -2) @ R
        if inv_quad_term.numel() and reduce_inv_quad:
            inv_quad_term = inv_quad_term.sum(-1)
        return inv_quad_term

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        if not self.is_square:
            raise RuntimeError(
                "inv_quad_logdet only operates on (batches of) square (positive semi-definite) LazyTensors. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if inv_quad_rhs is not None:
            if self.dim() == 2 and inv_quad_rhs.dim() == 1:
                if self.shape[-1] != inv_quad_rhs.numel():
                    raise RuntimeError(
                        "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                            self.shape, inv_quad_rhs.shape
                        )
                    )
            elif self.dim() != inv_quad_rhs.dim():
                raise RuntimeError(
                    "LazyTensor (size={}) and right-hand-side Tensor (size={}) should have the same number "
                    "of dimensions.".format(self.shape, inv_quad_rhs.shape)
                )
            elif self.shape[-1] != inv_quad_rhs.shape[-2]:
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, inv_quad_rhs.shape
                    )
                )

        inv_quad_term = None
        logdet_term = None

        if inv_quad_rhs is not None:
            inv_quad_term = self.inv_quad(inv_quad_rhs, reduce_inv_quad=reduce_inv_quad)

        if logdet:
            logdet_term = self._chol_diag.pow(2).log().sum(-1)

        return inv_quad_term, logdet_term
