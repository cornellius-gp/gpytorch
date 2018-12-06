#!/usr/bin/env python3

import torch

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

    def _getitem(self, *indices):
        has_added_tensor_index = False
        evaluate = False
        new_sizes = []

        for ix, sub_index in enumerate(indices):
            if isinstance(sub_index, int):
                if ix >= self.dim() - 2:
                    evaluate = True
                continue

            elif torch.is_tensor(sub_index):
                if ix >= self.dim() - 2:
                    evaluate = True
                if not has_added_tensor_index:
                    new_sizes.append(sub_index.numel())
                    has_added_tensor_index = True

            elif isinstance(sub_index, slice):
                new_sizes.append(len(torch.arange(self.size(ix))[sub_index]))
            else:
                new_sizes.append(len(sub_index))

        if evaluate:
            return torch.zeros(*new_sizes, dtype=self.dtype, device=self.device)
        else:
            return ZeroLazyTensor(*new_sizes)

    def _matmul(self, rhs):
        rhs_size_ind = -2 if rhs.ndimension() > 1 else -1
        if self.size(-1) != rhs.size(rhs_size_ind):
            raise RuntimeError("Size mismatch, self: {}, rhs: {}".format(self.size(), rhs.size()))
        return rhs * 0

    def _quad_form_derivative(self, left_vecs, right_vecs):
        raise RuntimeError("Backwards through a ZeroLazyTensor is not possible")

    def _size(self):
        return torch.Size(self.sizes)

    def _t_matmul(self, rhs):
        rhs_size_ind = -2 if rhs.ndimension() > 1 else -1
        if self.size(-1) != rhs.size(rhs_size_ind):
            raise RuntimeError("Size mismatch, self: {}, rhs: {}".format(self.size(), rhs.size()))
        return rhs * 0

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

    def inv_matmul(self, tensor):
        raise RuntimeError("ZeroLazyTensors are not invertible!")

    def inv_quad(self, tensor):
        raise RuntimeError("ZeroLazyTensors are not invertible!")

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False, reduce_inv_quad=True):
        raise RuntimeError("ZeroLazyTensors are not invertible!")

    def log_det(self):
        return torch.log(torch.tensor(0.0))

    def matmul(self, tensor):
        tensor_size_ind = -2 if tensor.ndimension() > 1 else -1
        if self.size(-1) != tensor.size(tensor_size_ind):
            raise RuntimeError("Size mismatch, self: {}, tensor: {}".format(self.size(), tensor.size()))
        return tensor * 0

    def mul(self, other):
        return self

    def mul_batch(self, mul_batch_size=None):
        return ZeroLazyTensor(*self.sizes[-2:])

    def root_decomposition(self):
        raise RuntimeError("ZeroLazyTensors are not positive definite!")

    def root_inv_decomposition(self, initial_vectors=None, test_vectors=None):
        raise RuntimeError("ZeroLazyTensors are not positive definite!")

    def root_decomposition_size(self):
        raise RuntimeError("ZeroLazyTensors are not positive definite!")

    def sum_batch(self, sum_batch_size=None):
        from .sum_batch.lazy_tensor import SumBatchLazyTensor

        return SumBatchLazyTensor(self, sum_batch_size=sum_batch_size)

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
