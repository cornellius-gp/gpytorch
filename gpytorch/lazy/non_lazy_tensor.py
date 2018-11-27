#!/usr/bin/env python3

import torch
from .. import settings
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

    def _matmul(self, rhs):
        return torch.matmul(self.tensor, rhs)

    def _t_matmul(self, rhs):
        return torch.matmul(self.tensor.transpose(-1, -2), rhs)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        res = left_vecs.matmul(right_vecs.transpose(-1, -2))
        return (res,)

    def _size(self):
        return self.tensor.size()

    def _transpose_nonbatch(self):
        return NonLazyTensor(self.tensor.transpose(-1, -2))

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        return self.tensor.__getitem__((*batch_indices, left_indices, right_indices))

    def diag(self):
        if self.tensor.ndimension() < 3:
            return self.tensor.diag()
        else:
            row_col_iter = torch.arange(0, self.matrix_shape[-1], dtype=torch.long, device=self.device)
            return self.tensor[..., row_col_iter, row_col_iter].view(*self.batch_shape, -1)

    def evaluate(self):
        return self.tensor

    def __getitem__(self, index):
        # Make sure index is a list
        if not isinstance(index, tuple):
            index = [index]
        else:
            index = list(index)

        # Handle the ellipsis
        # Find the index of the ellipsis
        ellipsis_locs = [index for index, item in enumerate(index) if item is Ellipsis]
        if settings.debug.on():
            if len(ellipsis_locs) > 1:
                raise RuntimeError(
                    "Cannot have multiple ellipsis in a __getitem__ call. LazyTensor {} "
                    " received index {}.".format(self, index)
                )
        if len(ellipsis_locs) == 1:
            ellipsis_loc = ellipsis_locs[0]
            num_to_fill_in = self.ndimension() - (len(index) - 1)
            index = index[:ellipsis_loc] + [slice(None, None, None)] * num_to_fill_in + index[ellipsis_loc + 1:]

        # Make the index a tuple again
        index = tuple(index)

        # Perform the __getitem__
        res = self.tensor[index]

        # Determine if we should return a LazyTensor or a Tensor
        if len(index) >= self.ndimension() - 1:
            row_index = index[self.ndimension() - 2]
            if isinstance(row_index, int) or torch.is_tensor(row_index):
                return res
        if len(index) == self.ndimension():
            col_index = index[self.ndimension() - 1]
            if isinstance(col_index, int) or torch.is_tensor(col_index):
                return res
        return NonLazyTensor(res)
