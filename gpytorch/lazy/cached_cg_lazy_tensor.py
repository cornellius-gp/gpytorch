#!/usr/bin/env python3

import torch
import warnings
from .lazy_tensor import LazyTensor
from .. import settings


class CachedCGLazyTensor(LazyTensor):
    """
    A LazyTensor wrapper that eagerly computes many CG calls in batch.
    This maximizes CG parallelism for fast inference.
    Used primarily for variational inference with GPs.

    Args:
        :attr:`base_lazy_tensor` (:class:`gpytorch.lazy.LazyTensor`): the LazyTensor to wrap
    """

    def __init__(self, base_lazy_tensor, eager_rhss=[], solves=None):
        # We're precomputing the solves and the normed version of the eager_rhss
        # This will make it faster when we reconstruct the LazyTensor inside functions
        with torch.no_grad():
            if solves is None:
                solves = [
                    base_lazy_tensor._solve(eager_rhs, base_lazy_tensor._preconditioner()[0])
                    for eager_rhs in eager_rhss
                ]

        super(CachedCGLazyTensor, self).__init__(
            base_lazy_tensor, eager_rhss=eager_rhss, solves=solves,
        )
        self.base_lazy_tensor = base_lazy_tensor
        self.eager_rhss = [eager_rhs.requires_grad_(False) for eager_rhs in eager_rhss]
        self.solves = [solve.requires_grad_(False) for solve in solves]

    @property
    def requires_grad(self):
        return self.base_lazy_tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, val):
        self.base_lazy_tensor.requires_grad = val

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        return self.base_lazy_tensor._get_indices(left_indices, right_indices, *batch_indices)

    def _getitem(self, *indices):
        return self.base_lazy_tensor._getitem(*indices)

    def _matmul(self, tensor):
        return self.base_lazy_tensor._matmul(tensor)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs)

    def _solve(self, rhs, preconditioner):
        # Here we check to see what solves we've already performed
        for eager_rhs, solve in zip(self.eager_rhss, self.solves):
            if torch.equal(rhs, eager_rhs):
                return solve

        if settings.debug.on():
            warnings.warn(
                "CachedCGLazyTensor had to run CG on a tensor of size {}. For best performance, this "
                "LazyTensor should pre-register all vectors to run CG against.".format(rhs.shape)
            )
        return super(CachedCGLazyTensor, self)._solve(rhs, preconditioner)

    def _size(self):
        return self.base_lazy_tensor._size()

    def _t_matmul(self, tensor):
        return self.base_lazy_tensor._t_matmul(tensor)

    def _transpose_nonbatch(self):
        return self.base_lazy_tensor._transpose_nonbatch()

    def detach_(self):
        self.base_lazy_tensor.detach_()
        return self
