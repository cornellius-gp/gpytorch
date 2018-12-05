#!/usr/bin/env python3

import torch
from ..lazy import LazyTensor


class NonLazyTensor(LazyTensor):
    def __init__(self, tsr):
        """
        Not a lazy tensor

        Args:
        - tsr (Tensor: matrix) a Tensor
        """
        if not torch.is_tensor(tsr):
            raise RuntimeError("NonLazyTensor must take a torch.Tensor; got {}".format(tsr.__class__.__name__))

        super(NonLazyTensor, self).__init__(tsr)
        self.tensor = tsr

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        return self.tensor.__getitem__((*batch_indices, left_indices, right_indices))

    def _getitem(self, *indices):
        # Perform the __getitem__
        res = self.tensor[indices]

        row_index = indices[-2]
        col_index = indices[-1]
        if (
            isinstance(row_index, int)
            or torch.is_tensor(row_index)
            or isinstance(col_index, int)
            or torch.is_tensor(col_index)
        ):
            return res
        return NonLazyTensor(res)

    def _matmul(self, rhs):
        return torch.matmul(self.tensor, rhs)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        res = left_vecs.matmul(right_vecs.transpose(-1, -2))
        return (res,)

    def _size(self):
        return self.tensor.size()

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
