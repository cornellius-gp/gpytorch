#!/usr/bin/env python3

import itertools
import torch

from .lazy_tensor import LazyTensor
from .root_lazy_tensor import RootLazyTensor


class BatchRepeatLazyTensor(LazyTensor):
    def __init__(self, base_lazy_tensor, batch_repeat=torch.Size((1,))):
        if not isinstance(batch_repeat, torch.Size):
            raise RuntimeError(
                "batch_repeat must be a torch.Size, got a {} instead".format(batch_repeat.__class__.__name__)
            )

        super(BatchRepeatLazyTensor, self).__init__(base_lazy_tensor, batch_repeat=batch_repeat)
        self.base_lazy_tensor = base_lazy_tensor
        self.batch_repeat = batch_repeat

    def _get_indices(self, row_indices, col_indices, *batch_indices):
        num_true_batch_dims = len(self.base_lazy_tensor.batch_shape)
        batch_indices = [index % size for index, size in zip(batch_indices, self._padded_base_batch_shape)]
        batch_indices = batch_indices[-num_true_batch_dims:] if num_true_batch_dims else []
        return self.base_lazy_tensor._get_indices(row_indices, col_indices, *batch_indices)

    def _getitem(self, *indices):
        args = []
        kwargs = self.base_lazy_tensor._kwargs
        num_base_batch_dims = len(self.base_lazy_tensor.batch_shape)

        for arg in self.base_lazy_tensor._args:
            if torch.is_tensor(arg):
                arg_base_shape_len = max(arg.dim() - num_base_batch_dims, 0)
                args.append(arg.repeat(*self.batch_repeat, *[1 for _ in range(arg_base_shape_len)]))
            elif isinstance(arg, LazyTensor):
                args.append(BatchRepeatLazyTensor(arg, batch_repeat=self.batch_repeat))
            else:
                args.append(arg)

        new_lazy_tensor = self.base_lazy_tensor.__class__(*args, **kwargs)
        return new_lazy_tensor._getitem(*indices)

    def _matmul(self, rhs):
        rhs = self._move_repeat_batches_to_columns(rhs)
        res = self.base_lazy_tensor._matmul(rhs)
        res = self._move_repeat_batches_back(res)
        return res

    def _move_repeat_batches_back(self, batch_matrix):
        """
        The opposite of _move_repeat_batches_to_columns

        Takes a b x m x nr tensor, and moves the batches associated with repeating
        So that the tensor is now rb x m x n.
        """
        orig_shape = (*self.batch_shape, batch_matrix.size(-2), -1)
        padded_base_batch_shape = self._padded_base_batch_shape

        # Now we have to move the columns back to their original repeat dimensions
        batch_matrix = batch_matrix.view(*padded_base_batch_shape, batch_matrix.size(-2), -1, *self.batch_repeat)
        dims = tuple(
            itertools.chain.from_iterable([i + len(orig_shape), i] for i in range(len(padded_base_batch_shape)))
        ) + (self.dim() - 2, self.dim() - 1)
        batch_matrix = batch_matrix.permute(*dims).contiguous()

        # Combine the repeat and the batch dimensions, and return the batch_matrixult!
        batch_matrix = batch_matrix.view(*orig_shape)
        return batch_matrix

    def _move_repeat_batches_to_columns(self, batch_matrix):
        """
        Takes a rb x m x n tensor, and moves the batches associated with repeating
        So that the tensor is now b x m x nr.
        This allows us to use the base_lazy_tensor routines.
        """
        batch_matrix_shape = batch_matrix.shape
        padded_base_batch_shape = self._padded_base_batch_shape

        # Reshape batch_matrix so that each batch dimension is split in two:
        # The repeated part, and the actual part
        split_shape = torch.Size(
            tuple(
                itertools.chain.from_iterable(
                    [repeat, size] for repeat, size in zip(self.batch_repeat, padded_base_batch_shape)
                )
            )
            + batch_matrix_shape[-2:]
        )
        batch_matrix = batch_matrix.view(*split_shape)

        # Now chuck the repeat parts of the batch dimensions into the last dimension of batch_matrix
        # These will act like extra columns of the batch matrix that we are multiplying against
        # The repeated part, and the actual part
        repeat_dims = range(0, len(self.batch_repeat) * 2, 2)
        batch_dims = range(1, len(self.batch_repeat) * 2, 2)
        batch_matrix = batch_matrix.permute(*batch_dims, -2, -1, *repeat_dims).contiguous()
        batch_matrix = batch_matrix.view(*self.base_lazy_tensor.batch_shape, batch_matrix_shape[-2], -1)
        return batch_matrix

    @property
    def _padded_base_batch_shape(self):
        base_batch_shape = self.base_lazy_tensor.batch_shape
        return torch.Size(([1] * (len(self.batch_repeat) - len(base_batch_shape))) + list(base_batch_shape))

    def _quad_form_derivative(self, left_vectors, right_vectors):
        left_vectors = self._move_repeat_batches_to_columns(left_vectors)
        right_vectors = self._move_repeat_batches_to_columns(right_vectors)
        return self.base_lazy_tensor._quad_form_derivative(left_vectors, right_vectors)

    def _size(self):
        repeated_batch_shape = torch.Size(
            size * repeat for size, repeat in zip(self._padded_base_batch_shape, self.batch_repeat)
        )
        res = torch.Size(repeated_batch_shape + self.base_lazy_tensor.matrix_shape)
        return res

    def _transpose_nonbatch(self):
        return self.__class__(self.base_lazy_tensor._transpose_nonbatch(), batch_repeat=self.batch_repeat)

    def add_jitter(self, jitter_val=1e-3):
        return self.__class__(self.base_lazy_tensor.add_jitter(jitter_val=jitter_val), batch_repeat=self.batch_repeat)

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False, reduce_inv_quad=True):
        if not self.is_square:
            raise RuntimeError(
                "inv_quad_log_det only operates on (batches of) square (positive semi-definite) LazyTensors. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if inv_quad_rhs is not None:
            if self.dim() != inv_quad_rhs.dim():
                raise RuntimeError(
                    "LazyTensor (size={}) and right-hand-side Tensor (size={}) should have the same number "
                    "of dimensions.".format(self.shape, inv_quad_rhs.shape)
                )
            elif self.batch_shape != inv_quad_rhs.shape[:-2] or self.shape[-1] != inv_quad_rhs.shape[-2]:
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, inv_quad_rhs.shape
                    )
                )

        if inv_quad_rhs is not None:
            inv_quad_rhs = self._move_repeat_batches_to_columns(inv_quad_rhs)

        inv_quad_term, log_det_term = self.base_lazy_tensor.inv_quad_log_det(
            inv_quad_rhs, log_det, reduce_inv_quad=False
        )

        if inv_quad_term is not None and inv_quad_term.numel():
            inv_quad_term = inv_quad_term.view(*inv_quad_term.shape[:-1], -1, self.batch_repeat.numel())
            inv_quad_term = self._move_repeat_batches_back(inv_quad_term).squeeze(-1)
            if reduce_inv_quad:
                inv_quad_term = inv_quad_term.sum(-1)

        if log_det_term is not None and log_det_term.numel():
            log_det_term = log_det_term.repeat(*self.batch_repeat)

        return inv_quad_term, log_det_term

    def repeat(self, *sizes):
        if len(sizes) < 3 or tuple(sizes[-2:]) != (1, 1):
            raise RuntimeError(
                "Invalid repeat arguments {}. Currently, repeat only works to create repeated "
                "batches of a 2D LazyTensor.".format(tuple(sizes))
            )

        padded_batch_repeat = tuple(1 for _ in range(len(sizes) - 2 - len(self.batch_repeat))) + self.batch_repeat
        return self.__class__(
            self,
            batch_repeat=torch.Size(
                orig_repeat_size * new_repeat_size
                for orig_repeat_size, new_repeat_size in zip(padded_batch_repeat, sizes[:-2])
            ),
        )

    def root_decomposition(self):
        return RootLazyTensor(self.base_lazy_tensor.root_decomposition().root.repeat(*self.batch_repeat, 1, 1))

    def root_inv_decomposition(self, initial_vectors=None, test_vectors=None):
        return RootLazyTensor(
            self.base_lazy_tensor.root_inv_decomposition(
                initial_vectors=initial_vectors, test_vectors=test_vectors
            ).root.repeat(*self.batch_repeat, 1, 1)
        )
