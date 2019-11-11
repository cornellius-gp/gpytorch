#!/usr/bin/env python3

import torch

from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.getitem import _compute_getitem_size
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor


class ZeroLazyTensor(LazyTensor):
    """
    Special LazyTensor representing zero.
    """

    def __init__(self, *sizes, dtype=None, device=None):
        super(ZeroLazyTensor, self).__init__(*sizes)
        self.sizes = list(sizes)

        self._dtype = dtype or torch.get_default_dtype()
        self._device = device or torch.device("cpu")

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def _expand_batch(self, batch_shape):
        return self.__class__(*batch_shape, *self.sizes[-2:], dtype=self._dtype, device=self._device)

    def _get_indices(self, row_index, col_index, *batch_indices):
        new_size = _compute_getitem_size(self, batch_indices + (row_index, col_index))
        return ZeroLazyTensor(*new_size)

    def _getitem(self, row_index, col_index, *batch_indices):
        new_size = _compute_getitem_size(self, batch_indices + (row_index, col_index))
        return ZeroLazyTensor(*new_size)

    def _matmul(self, rhs):
        rhs_size_ind = -2 if rhs.ndimension() > 1 else -1
        if self.size(-1) != rhs.size(rhs_size_ind):
            raise RuntimeError("Size mismatch, self: {}, rhs: {}".format(self.size(), rhs.size()))
        return rhs * 0

    def _prod_batch(self, dim):
        sizes = list(self.sizes)
        del sizes[dim]
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        raise RuntimeError("Backwards through a ZeroLazyTensor is not possible")

    def _root_decomposition(self):
        raise RuntimeError("ZeroLazyTensors are not positive definite!")

    def _root_inv_decomposition(self, initial_vectors=None):
        raise RuntimeError("ZeroLazyTensors are not positive definite!")

    def _root_decomposition_size(self):
        raise RuntimeError("ZeroLazyTensors are not positive definite!")

    def _size(self):
        return torch.Size(self.sizes)

    def _sum_batch(self, dim):
        sizes = list(self.sizes)
        del sizes[dim]
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def _t_matmul(self, rhs):
        rhs_size_ind = -2 if rhs.ndimension() > 1 else -1
        if self.size(-1) != rhs.size(rhs_size_ind):
            raise RuntimeError("Size mismatch, self: {}, rhs: {}".format(self.size(), rhs.size()))
        return rhs * 0

    def _transpose_nonbatch(self):
        return self.transpose(-2, -1)

    def _unsqueeze_batch(self, dim):
        sizes = self.sizes.copy()
        sizes.insert(dim, 1)
        return self.__class__(*sizes, dtype=self._dtype, device=self._device)

    def add_diag(self, diag):
        from .diag_lazy_tensor import DiagLazyTensor

        if self.size(-1) != self.size(-2):
            raise RuntimeError("add_diag only defined for square matrices")

        if self.ndimension() == 3:
            if diag.ndimension() == 0:
                diag = diag.view(1, 1).expand(self.size(0), self.size(1))
            elif diag.ndimension() == 1:
                diag = diag.unsqueeze(0).expand(self.size(0), self.size(1))
            elif diag.ndimension() == 2:
                diag = diag.expand(self.size(0), self.size(1))
            else:
                raise RuntimeError(
                    "For a 3D tensor ({}), add_diag expects a 1D or 2D diag. "
                    "Got size ({})".format(self.size(), diag.size())
                )
        else:
            if diag.ndimension() == 0:
                diag = diag.view(1).expand(self.size(0))
            elif diag.ndimension() == 1:
                diag = diag.expand(self.size(0))
            else:
                raise RuntimeError(
                    "For a 3D tensor ({}), add_diag expects a 1D or 2D diag. "
                    "Got size ({})".format(self.size(), diag.size())
                )

        res = DiagLazyTensor(diag)
        if res.size() != self.size():
            raise RuntimeError(
                "Diag dimensions are incompatible with the base LazyTensor dimensions. "
                "Diag size corresponds to a {} Tensor - expected {}".format(res.size(), self.size())
            )
        return res

    def diag(self):
        shape = self.shape
        if shape[-1] != shape[-2]:
            raise RuntimeError("diag works on square matrices (or batches)")
        return torch.zeros(shape[:-1], dtype=self.dtype, device=self.device)

    @cached
    def evaluate(self):
        return torch.zeros(*self.sizes)

    def inv_matmul(self, right_tensor, left_tensor=None):
        raise RuntimeError("ZeroLazyTensors are not invertible!")

    def inv_quad(self, tensor):
        raise RuntimeError("ZeroLazyTensors are not invertible!")

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        raise RuntimeError("ZeroLazyTensors are not invertible!")

    def logdet(self):
        return torch.log(torch.tensor(0.0))

    def matmul(self, tensor):
        tensor_size_ind = -2 if tensor.ndimension() > 1 else -1
        if self.size(-1) != tensor.size(tensor_size_ind):
            raise RuntimeError("Size mismatch, self: {}, tensor: {}".format(self.size(), tensor.size()))
        return tensor * 0

    def mul(self, other):
        shape = _mul_broadcast_shape(self.shape, other.shape)
        return self.__class__(*shape, dtype=self._dtype, device=self._device)

    def transpose(self, dim1, dim2):
        sizes = self.sizes.copy()
        tmp = sizes[dim1]
        sizes[dim1] = sizes[dim2]
        sizes[dim2] = tmp

        return ZeroLazyTensor(*sizes)

    def __add__(self, other):
        return other

    def __div__(self, other):
        return self

    def __mul__(self, other):
        return self
