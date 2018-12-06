#!/usr/bin/env python3

from itertools import product

import torch

from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor
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

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        """Extract elements from the LazyTensor. Supports arbitrary batch sizes.

        Args:
            :attr:`left_indices` (LongTensor): An `i`-dim tensor of row indices (these are the
                same across all batches).
            :attr:`left_indices` (LongTensor): An `i`-dim tensor of column indices (these are the
                same across all batches).
            :attr:`batch_indices` (LongTensor): A variable number of `i`-dim tensors of batch indices,
                one for each batch dimension. If smaller than the number of batch dimensions of the
                DiagLazyTensor, select all elements of the omitted batches.

        Returns:
            An `i`-dim tensor containing the requested elements if all batch index tensors were
            passed in, otherwise a `bl x ... bk x i` batch of such tensors, where `l = len(batch_indices) + 1`.

        """
        batch_dims = len(batch_indices)
        if batch_dims > self.batch_dim:
            raise RuntimeError(
                "Received {} batch index tensors, DiagLazyTensor has {} batches".format(batch_dims, self.batch_dim)
            )
        full_batch_shape = self._diag.shape[batch_dims:-1]
        if len(full_batch_shape) == 0:
            diag_elements = self._diag[batch_indices + (left_indices,)]
        else:
            d = torch.ones_like(left_indices)
            diag_elements = torch.stack(
                [
                    self._diag[batch_indices + tuple(j * d for j in js) + (left_indices,)]
                    for js in product(*[range(b) for b in full_batch_shape])
                ]
            )
        equal_indices = (left_indices == right_indices).type_as(self._diag)
        return diag_elements * equal_indices

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
        if torch.is_tensor(other):
            other = NonLazyTensor(other)

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

    def inv_matmul(self, tensor):
        return self.inverse()._matmul(tensor)

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False, reduce_inv_quad=True):

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

        if not log_det:
            log_det_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            log_det_term = self._diag.log().sum(-1)

        return inv_quad_term, log_det_term

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

    def sum_batch(self, sum_batch_size=None):
        if sum_batch_size is not None:
            if self.batch_dim > 1:
                raise NotImplementedError("batch dimensions > 1 not yet supported with sum_batch_size")
            return DiagLazyTensor(self._diag.view(-1, sum_batch_size, self._diag.shape[-1]))
        return DiagLazyTensor(self._diag.sum([i for i in range(self.batch_dim)]))

    def zero_mean_mvn_samples(self, num_samples):
        base_samples = torch.randn(num_samples, *self._diag.shape, dtype=self.dtype, device=self.device)
        return base_samples * self._diag.sqrt()
