#!/usr/bin/env python3

import torch

from .lazy_tensor import LazyTensor


class NonLazyTensor(LazyTensor):
    def _check_args(self, tsr):
        if not torch.is_tensor(tsr):
            return "NonLazyTensor must take a torch.Tensor; got {}".format(tsr.__class__.__name__)
        if tsr.dim() < 2:
            return "NonLazyTensor expects a matrix (or batches of matrices) - got a Tensor of size {}.".format(
                tsr.shape
            )

    def __init__(self, tsr):
        """
        Not a lazy tensor

        Args:
        - tsr (Tensor: matrix) a Tensor
        """
        super(NonLazyTensor, self).__init__(tsr)
        self.tensor = tsr

    def _cholesky_solve(self, rhs, upper=False):
        return torch.cholesky_solve(rhs, self.evaluate(), upper=upper)

    def _expand_batch(self, batch_shape):
        return self.__class__(self.tensor.expand(*batch_shape, *self.matrix_shape))

    def _get_indices(self, row_index, col_index, *batch_indices):
        # Perform the __getitem__
        res = self.tensor[(*batch_indices, row_index, col_index)]
        return res

    def _getitem(self, row_index, col_index, *batch_indices):
        # Perform the __getitem__
        res = self.tensor[(*batch_indices, row_index, col_index)]
        return self.__class__(res)

    def _matmul(self, rhs):
        return torch.matmul(self.tensor, rhs)

    def _prod_batch(self, dim):
        return self.__class__(self.tensor.prod(dim))

    def _quad_form_derivative(self, left_vecs, right_vecs):
        res = left_vecs.matmul(right_vecs.transpose(-1, -2))
        return (res,)

    def _size(self):
        return self.tensor.size()

    def _sum_batch(self, dim):
        return self.__class__(self.tensor.sum(dim))

    def _transpose_nonbatch(self):
        return NonLazyTensor(self.tensor.transpose(-1, -2))

    def _t_matmul(self, rhs):
        return torch.matmul(self.tensor.transpose(-1, -2), rhs)

    def diag(self):
        if self.tensor.ndimension() < 3:
            return self.tensor.diag()
        else:
            row_col_iter = torch.arange(0, self.matrix_shape[-1], dtype=torch.long, device=self.device)
            return self.tensor[..., row_col_iter, row_col_iter].view(*self.batch_shape, -1)

    def evaluate(self):
        return self.tensor

    def __add__(self, other):
        if isinstance(other, NonLazyTensor):
            return NonLazyTensor(self.tensor + other.tensor)
        elif isinstance(other, torch.Tensor):
            return NonLazyTensor(self.tensor + other)
        else:
            return super(NonLazyTensor, self).__add__(other)

    def mul(self, other):
        if isinstance(other, NonLazyTensor):
            return NonLazyTensor(self.tensor * other.tensor)
        else:
            return super(NonLazyTensor, self).mul(other)


def lazify(obj):
    """
    A function which ensures that `obj` is a LazyTensor.

    If `obj` is a LazyTensor, this function does nothing.
    If `obj` is a (normal) Tensor, this function wraps it with a `NonLazyTensor`.
    """

    if torch.is_tensor(obj):
        return NonLazyTensor(obj)
    elif isinstance(obj, LazyTensor):
        return obj
    else:
        raise TypeError("object of class {} cannot be made into a LazyTensor".format(obj.__class__.__name__))


__all__ = ["NonLazyTensor", "lazify"]
