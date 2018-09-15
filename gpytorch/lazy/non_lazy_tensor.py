from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from ..functions import add_diag
from ..lazy import LazyTensor


class NonLazyTensor(LazyTensor):
    def __init__(self, tsr):
        """
        Not a lazy tensor

        Args:
        - tsr (Tensor: matrix) a Tensor
        """
        if not torch.is_tensor(tsr):
            raise RuntimeError(
                "NonLazyTensor must take a torch.Tensor; got {}".format(tsr.__class__.__name__)
            )

        super(NonLazyTensor, self).__init__(tsr)
        self.tensor = tsr

    def _matmul(self, rhs):
        return torch.matmul(self.tensor, rhs)

    def _t_matmul(self, rhs):
        return torch.matmul(self.tensor.transpose(-1, -2), rhs)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() < self.tensor.ndimension():
            left_vecs = left_vecs.unsqueeze(-1)
            right_vecs = right_vecs.unsqueeze(-1)

        res = left_vecs.matmul(right_vecs.transpose(-1, -2))
        return (res,)

    def _size(self):
        return self.tensor.size()

    def _transpose_nonbatch(self):
        return NonLazyTensor(self.tensor.transpose(-1, -2))

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        return self.tensor[batch_indices.data, left_indices.data, right_indices.data]

    def _get_indices(self, left_indices, right_indices):
        return self.tensor[left_indices.data, right_indices.data]

    def add_diag(self, diag):
        return NonLazyTensor(add_diag(self.tensor, diag))

    def _preconditioner(self):
        # For a NonLazyTensor, it is intended to not use preconditioning, even when called for.
        return None, None

    def diag(self):
        if self.tensor.ndimension() < 3:
            return self.tensor.diag()
        else:
            size = self.size()
            row_col_iter = self.tensor_cls(size[-1]).long()
            torch.arange(0, size[-1], out=row_col_iter)
            batch_iter = self.tensor_cls(size[0]).long()
            torch.arange(0, size[0], out=batch_iter)
            batch_iter = batch_iter.unsqueeze(1).repeat(1, size[1]).view(-1)
            row_col_iter = row_col_iter.unsqueeze(1).repeat(size[0], 1).view(-1)
            return self.tensor[batch_iter, row_col_iter, row_col_iter].view(size[0], size[1])

    def evaluate(self):
        return self.tensor

    def repeat(self, *sizes):
        return NonLazyTensor(self.tensor.repeat(*sizes))

    def __getitem__(self, index):
        return NonLazyTensor(self.tensor[index])
