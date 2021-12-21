#!/usr/bin/env python3

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.getitem import _compute_getitem_size, _is_noop_index
from ..utils.memoize import cached
from .diag_lazy_tensor import ConstantDiagLazyTensor
from .lazy_tensor import LazyTensor
from .zero_lazy_tensor import ZeroLazyTensor


class IdentityLazyTensor(ConstantDiagLazyTensor):
    def __init__(self, diag_shape, batch_shape=torch.Size([]), dtype=None, device=None):
        """
        Identity matrix lazy tensor. Supports arbitrary batch sizes.

        Args:
            :attr:`diag` (Tensor):
                A `b1 x ... x bk x n` Tensor, representing a `b1 x ... x bk`-sized batch
                of `n x n` identity matrices
        """
        one = torch.tensor(1.0, dtype=dtype, device=device)
        LazyTensor.__init__(self, diag_shape=diag_shape, batch_shape=batch_shape, dtype=dtype, device=device)
        self.diag_values = one.expand(torch.Size([*batch_shape, 1]))
        self.diag_shape = diag_shape
        self._batch_shape = batch_shape
        self._dtype = dtype
        self._device = device

    @property
    def batch_shape(self):
        """
        Returns the shape over which the tensor is batched.
        """
        return self._batch_shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def _maybe_reshape_rhs(self, rhs):
        if self._batch_shape != rhs.shape[:-2]:
            batch_shape = _mul_broadcast_shape(rhs.shape[:-2], self._batch_shape)
            return rhs.expand(*batch_shape, *rhs.shape[-2:])
        else:
            return rhs

    @cached(name="cholesky", ignore_args=True)
    def _cholesky(self, upper=False):
        return self

    def _cholesky_solve(self, rhs):
        return self._maybe_reshape_rhs(rhs)

    def _expand_batch(self, batch_shape):
        return IdentityLazyTensor(
            diag_shape=self.diag_shape, batch_shape=batch_shape, dtype=self.dtype, device=self.device
        )

    def _getitem(self, row_index, col_index, *batch_indices):
        # Special case: if both row and col are not indexed, then we are done
        if _is_noop_index(row_index) and _is_noop_index(col_index):
            if len(batch_indices):
                new_batch_shape = _compute_getitem_size(self, (*batch_indices, row_index, col_index))[:-2]
                res = IdentityLazyTensor(
                    diag_shape=self.diag_shape, batch_shape=new_batch_shape, dtype=self._dtype, device=self._device
                )
                return res
            else:
                return self

        return super()._getitem(row_index, col_index, *batch_indices)

    def _matmul(self, rhs):
        return self._maybe_reshape_rhs(rhs)

    def _mul_constant(self, constant):
        return ConstantDiagLazyTensor(self.diag_values * constant, diag_shape=self.diag_shape)

    def _mul_matrix(self, other):
        return other

    def _permute_batch(self, *dims):
        batch_shape = self.diag_values.permute(*dims, -1).shape[:-1]
        return IdentityLazyTensor(
            diag_shape=self.diag_shape, batch_shape=batch_shape, dtype=self._dtype, device=self._device
        )

    def _prod_batch(self, dim):
        batch_shape = list(self.batch_shape)
        del batch_shape[dim]
        return IdentityLazyTensor(
            diag_shape=self.diag_shape, batch_shape=torch.Size(batch_shape), dtype=self.dtype, device=self.device
        )

    def _root_decomposition(self):
        return self.sqrt()

    def _root_inv_decomposition(self, initial_vectors=None):
        return self.inverse().sqrt()

    def _size(self):
        return torch.Size([*self._batch_shape, self.diag_shape, self.diag_shape])

    def _t_matmul(self, rhs):
        return self._maybe_reshape_rhs(rhs)

    def _transpose_nonbatch(self):
        return self

    def abs(self):
        return self

    def exp(self):
        return self

    def inverse(self):
        return self

    def inv_matmul(self, right_tensor, left_tensor=None):
        res = self._maybe_reshape_rhs(right_tensor)
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        # TODO: Use proper batching for inv_quad_rhs (prepand to shape rathern than append)
        if inv_quad_rhs is None:
            inv_quad_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            rhs_batch_shape = inv_quad_rhs.shape[1 + self.batch_dim :]
            inv_quad_term = inv_quad_rhs.mul(inv_quad_rhs).sum(-(1 + len(rhs_batch_shape)))
            if reduce_inv_quad:
                inv_quad_term = inv_quad_term.sum(-1)

        if logdet:
            logdet_term = torch.zeros(self.batch_shape, dtype=self.dtype, device=self.device)
        else:
            logdet_term = torch.empty(0, dtype=self.dtype, device=self.device)

        return inv_quad_term, logdet_term

    def log(self):
        return ZeroLazyTensor(
            *self._batch_shape, self.diag_shape, self.diag_shape, dtype=self._dtype, device=self._device
        )

    def matmul(self, other):
        is_vec = False
        if other.dim() == 1:
            is_vec = True
            other = other.unsqueeze(-1)
        res = self._maybe_reshape_rhs(other)
        if is_vec:
            res = res.squeeze(-1)
        return res

    def sqrt(self):
        return self

    def sqrt_inv_matmul(self, rhs, lhs=None):
        if lhs is None:
            return self._maybe_reshape_rhs(rhs)
        else:
            sqrt_inv_matmul = lhs @ rhs
            inv_quad = lhs.pow(2).sum(dim=-1)
            return sqrt_inv_matmul, inv_quad

    def type(self, dtype):
        """
        This method operates similarly to :func:`torch.Tensor.type`.
        """
        return IdentityLazyTensor(
            diag_shape=self.diag_shape, batch_shape=self.batch_shape, dtype=dtype, device=self.device
        )

    def zero_mean_mvn_samples(self, num_samples):
        base_samples = torch.randn(num_samples, *self.shape[:-1], dtype=self.dtype, device=self.device)
        return base_samples

    @cached(name="svd")
    def _svd(self) -> Tuple[LazyTensor, Tensor, LazyTensor]:
        return self, self._diag, self

    def _symeig(self, eigenvectors: bool = False) -> Tuple[Tensor, Optional[LazyTensor]]:
        return self._diag, self
