from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import product
import torch
from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .lazy_tensor import LazyTensor
from ..utils.broadcasting import _mul_broadcast_shape


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
        self._batch_shape = diag.shape[:-1]

    def __add__(self, other):
        if isinstance(other, DiagLazyTensor):
            return self.add_diag(other._diag)
        return AddedDiagLazyTensor(other, self)

    def _get_indices(self, indices):
        """Extract elements from the LazyTensor. Supports arbitrary batch sizes and broadcasting across batch sizes.

        Args:
            :attr:`indices` (LongTensor):
                A `b1 x ... bk x 2 x j` LongTensor, for each (multi-dimensional) batch selecting j
                elements from the associated `n x n` diagonal matrix. The `2 x j` tensor `t` associated
                with the last two dimensions represents the indices as lists of row and column indices,
                that is, selecting the `b1 x ... x bk x t(0, l) x t(1, l)`-th element.
                The batch indices follow standard broadcasting rules.

        """
        batch_shape = _mul_broadcast_shape(indices.shape[:-2], self._diag.shape[:-1])
        row_indices = indices[..., 0, :].expand(batch_shape + indices.shape[-1:])
        diag = self._diag.expand(batch_shape + self._diag.shape[-1:])

        diag_elements = torch.stack(
            [diag[batch_idx][row_indices[batch_idx]] for batch_idx in product(*(range(i) for i in batch_shape))]
        ).view(batch_shape + indices.shape[-1:])

        equal_indices = (indices[..., -1, :] == indices[..., -2, :]).type_as(self._diag)
        return diag_elements * equal_indices

    def _matmul(self, rhs):
        # this are trivial if we multiply two DiagLazyTensors
        if isinstance(rhs, DiagLazyTensor):
            return DiagLazyTensor(self._diag * rhs._diag)
        # to perform matrix multiplication with diagonal matrices we can just
        # multiply element-wise with the diagoanl using proper broadcasting
        return self._diag.unsqueeze(-1) * rhs

    def _quad_form_derivative(self, left_vecs, right_vecs):
        res = left_vecs * right_vecs
        batch_shape = _mul_broadcast_shape(self._batch_shape, res.shape[:-1])
        if batch_shape != res.shape[:-1]:
            res = res.expand(batch_shape + res.shape[-1:])
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
        return DiagLazyTensor(self._diag + added_diag)

    def diag(self):
        return self._diag

    def evaluate(self):
        return self._diag.unsqueeze(-1) * torch.eye(self._diag.shape[-1])

    def exp(self):
        return DiagLazyTensor(self._diag.exp())

    def inverse(self):
        return DiagLazyTensor(self._diag.reciprocal())

    def inv_matmul(self, tensor):
        return self.inverse()._matmul(tensor)

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False, reduce_inv_quad=True):

        if inv_quad_rhs is None:
            inv_quad_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            if self._diag.shape[-1] != inv_quad_rhs.shape[-1]:
                raise Exception
            inv_quad_term = inv_quad_rhs.div(self._diag).mul(inv_quad_rhs).sum(-1)
            if reduce_inv_quad:
                inv_quad_term = inv_quad_term.sum(-1)

        if not log_det:
            log_det_term = torch.empty(0, dtype=self.dtype, device=self.device)
        else:
            log_det_term = self._diag.log().sum(-1)

        return inv_quad_term, log_det_term

    def log(self):
        return DiagLazyTensor(self._diag.log())

    def root_decomposition(self):
        return DiagLazyTensor(self._diag.sqrt())

    def root_inv_decomposition(self, initial_vectors=None, test_vectors=None):
        return self.root_decomposition().inverse()

    def sum_batch(self, sum_batch_size=None):
        raise NotImplementedError

    def zero_mean_mvn_samples(self, num_samples):
        base_samples = torch.randn(num_samples, *self._diag.shape, dtype=self.dtype, device=self.device)
        return base_samples * self._diag.sqrt()
