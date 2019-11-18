#!/usr/bin/env python3

import torch

from ..utils import sparse
from ..utils.broadcasting import _pad_with_singletons
from ..utils.getitem import _noop_index
from ..utils.interpolation import left_interp, left_t_interp
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor, lazify
from .root_lazy_tensor import RootLazyTensor


class InterpolatedLazyTensor(LazyTensor):
    def _check_args(
        self, base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
    ):
        if left_interp_indices.size() != left_interp_values.size():
            return "Expected left_interp_indices ({}) to have the same size as left_interp_values ({})".format(
                left_interp_indices.size(), left_interp_values.size()
            )
        if right_interp_indices.size() != right_interp_values.size():
            return "Expected right_interp_indices ({}) to have the same size as right_interp_values ({})".format(
                right_interp_indices.size(), right_interp_values.size()
            )
        if left_interp_indices.shape[:-2] != right_interp_indices.shape[:-2]:
            return (
                "left interp size ({}) is incompatible with right interp size ({}). Make sure the two have the "
                "same number of batch dimensions".format(left_interp_indices.size(), right_interp_indices.size())
            )
        if left_interp_indices.shape[:-2] != base_lazy_tensor.shape[:-2]:
            return (
                "left interp size ({}) is incompatible with base lazy tensor size ({}). Make sure the two have the "
                "same number of batch dimensions".format(left_interp_indices.size(), base_lazy_tensor.size())
            )

    def __init__(
        self,
        base_lazy_tensor,
        left_interp_indices=None,
        left_interp_values=None,
        right_interp_indices=None,
        right_interp_values=None,
    ):
        base_lazy_tensor = lazify(base_lazy_tensor)

        if left_interp_indices is None:
            num_rows = base_lazy_tensor.size(-2)
            left_interp_indices = torch.arange(0, num_rows, dtype=torch.long, device=base_lazy_tensor.device)
            left_interp_indices.unsqueeze_(-1)
            left_interp_indices = left_interp_indices.expand(*base_lazy_tensor.batch_shape, num_rows, 1)

        if left_interp_values is None:
            left_interp_values = torch.ones(
                left_interp_indices.size(), dtype=base_lazy_tensor.dtype, device=base_lazy_tensor.device
            )

        if right_interp_indices is None:
            num_cols = base_lazy_tensor.size(-1)
            right_interp_indices = torch.arange(0, num_cols, dtype=torch.long, device=base_lazy_tensor.device)
            right_interp_indices.unsqueeze_(-1)
            right_interp_indices = right_interp_indices.expand(*base_lazy_tensor.batch_shape, num_cols, 1)

        if right_interp_values is None:
            right_interp_values = torch.ones(
                right_interp_indices.size(), dtype=base_lazy_tensor.dtype, device=base_lazy_tensor.device
            )

        if left_interp_indices.shape[:-2] != base_lazy_tensor.batch_shape:
            try:
                base_lazy_tensor = base_lazy_tensor._expand_batch(left_interp_indices.shape[:-2])
            except RuntimeError:
                raise RuntimeError(
                    "interp size ({}) is incompatible with base_lazy_tensor size ({}). ".format(
                        right_interp_indices.size(), base_lazy_tensor.size()
                    )
                )

        super(InterpolatedLazyTensor, self).__init__(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        self.base_lazy_tensor = base_lazy_tensor
        self.left_interp_indices = left_interp_indices
        self.left_interp_values = left_interp_values
        self.right_interp_indices = right_interp_indices
        self.right_interp_values = right_interp_values

    def _approx_diag(self):
        base_diag_root = self.base_lazy_tensor.diag().sqrt()
        left_res = left_interp(self.left_interp_indices, self.left_interp_values, base_diag_root.unsqueeze(-1))
        right_res = left_interp(self.right_interp_indices, self.right_interp_values, base_diag_root.unsqueeze(-1))
        res = left_res * right_res
        return res.squeeze(-1)

    def _expand_batch(self, batch_shape):
        return self.__class__(
            self.base_lazy_tensor._expand_batch(batch_shape),
            self.left_interp_indices.expand(*batch_shape, *self.left_interp_indices.shape[-2:]),
            self.left_interp_values.expand(*batch_shape, *self.left_interp_values.shape[-2:]),
            self.right_interp_indices.expand(*batch_shape, *self.right_interp_indices.shape[-2:]),
            self.right_interp_values.expand(*batch_shape, *self.right_interp_values.shape[-2:]),
        )

    def _get_indices(self, row_index, col_index, *batch_indices):
        left_interp_indices = self.left_interp_indices.__getitem__((*batch_indices, row_index)).unsqueeze(-2)
        right_interp_indices = self.right_interp_indices.__getitem__((*batch_indices, col_index)).unsqueeze(-1)
        base_vals = self.base_lazy_tensor._get_indices(
            left_interp_indices,
            right_interp_indices,
            *[batch_index.view(*batch_index.shape, 1, 1) for batch_index in batch_indices],
        )

        left_interp_values = self.left_interp_values.__getitem__((*batch_indices, row_index)).unsqueeze(-2)
        right_interp_values = self.right_interp_values.__getitem__((*batch_indices, col_index)).unsqueeze(-1)
        interp_values = left_interp_values * right_interp_values

        res = (base_vals * interp_values).sum([-2, -1])
        return res

    def _getitem(self, row_index, col_index, *batch_indices):
        # Handle batch dimensions
        # Construt a new LazyTensor
        base_lazy_tensor = self.base_lazy_tensor
        left_interp_indices = self.left_interp_indices
        left_interp_values = self.left_interp_values
        right_interp_indices = self.right_interp_indices
        right_interp_values = self.right_interp_values

        if len(batch_indices):
            base_lazy_tensor = base_lazy_tensor._getitem(_noop_index, _noop_index, *batch_indices)

        # Special case: if both row and col are not indexed, then we are done
        if row_index is _noop_index and col_index is _noop_index:
            left_interp_indices = left_interp_indices[batch_indices]
            left_interp_values = left_interp_values[batch_indices]
            right_interp_indices = right_interp_indices[batch_indices]
            right_interp_values = right_interp_values[batch_indices]

            return self.__class__(
                base_lazy_tensor,
                left_interp_indices,
                left_interp_values,
                right_interp_indices,
                right_interp_values,
                **self._kwargs,
            )

        # Normal case: we have to do some processing on either the rows or columns
        # We will handle this through "interpolation"
        left_interp_indices = left_interp_indices[(*batch_indices, row_index, _noop_index)]
        left_interp_values = left_interp_values[(*batch_indices, row_index, _noop_index)]
        right_interp_indices = right_interp_indices[(*batch_indices, col_index, _noop_index)]
        right_interp_values = right_interp_values[(*batch_indices, col_index, _noop_index)]

        # Construct interpolated LazyTensor
        res = self.__class__(
            base_lazy_tensor,
            left_interp_indices,
            left_interp_values,
            right_interp_indices,
            right_interp_values,
            **self._kwargs,
        )
        return res

    def _matmul(self, rhs):
        # Get sparse tensor representations of left/right interp matrices
        left_interp_t = self._sparse_left_interp_t(self.left_interp_indices, self.left_interp_values)
        right_interp_t = self._sparse_right_interp_t(self.right_interp_indices, self.right_interp_values)

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        # right_interp^T * rhs
        right_interp_res = sparse.bdsmm(right_interp_t, rhs)

        # base_lazy_tensor * right_interp^T * rhs
        base_res = self.base_lazy_tensor._matmul(right_interp_res)

        # left_interp * base_lazy_tensor * right_interp^T * rhs
        left_interp_mat = left_interp_t.transpose(-1, -2)
        res = sparse.bdsmm(left_interp_mat, base_res)

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    def _mul_constant(self, other):
        # We're using a custom method here - the constant mul is applied to the base_lazy tensor
        # This preserves the interpolated structure
        return self.__class__(
            self.base_lazy_tensor._mul_constant(other),
            self.left_interp_indices,
            self.left_interp_values,
            self.right_interp_indices,
            self.right_interp_values,
        )

    def _t_matmul(self, rhs):
        # Get sparse tensor representations of left/right interp matrices
        left_interp_t = self._sparse_left_interp_t(self.left_interp_indices, self.left_interp_values)
        right_interp_t = self._sparse_right_interp_t(self.right_interp_indices, self.right_interp_values)

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        # right_interp^T * rhs
        left_interp_res = sparse.bdsmm(left_interp_t, rhs)

        # base_lazy_tensor * right_interp^T * rhs
        base_res = self.base_lazy_tensor._t_matmul(left_interp_res)

        # left_interp * base_lazy_tensor * right_interp^T * rhs
        right_interp_mat = right_interp_t.transpose(-1, -2)
        res = sparse.bdsmm(right_interp_mat, base_res)

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        # Get sparse tensor representations of left/right interp matrices
        left_interp_t = self._sparse_left_interp_t(self.left_interp_indices, self.left_interp_values)
        right_interp_t = self._sparse_right_interp_t(self.right_interp_indices, self.right_interp_values)

        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)

        # base_lazy_tensor grad
        left_res = sparse.bdsmm(left_interp_t, left_vecs)
        right_res = sparse.bdsmm(right_interp_t, right_vecs)
        base_lv_grad = list(self.base_lazy_tensor._quad_form_derivative(left_res, right_res))

        # left_interp_values grad
        n_vecs = right_res.size(-1)
        n_left_rows = self.left_interp_indices.size(-2)
        n_right_rows = self.right_interp_indices.size(-2)
        n_left_interp = self.left_interp_indices.size(-1)
        n_right_interp = self.right_interp_indices.size(-1)
        n_inducing = right_res.size(-2)

        # left_interp_values grad
        right_interp_right_res = self.base_lazy_tensor._matmul(right_res).contiguous()
        batch_shape = torch.Size(right_interp_right_res.shape[:-2])
        batch_size = batch_shape.numel()
        if len(batch_shape):
            batch_offset = torch.arange(0, batch_size, dtype=torch.long, device=self.device).view(*batch_shape)
            batch_offset.unsqueeze_(-1).unsqueeze_(-1).mul_(n_inducing)
            batched_right_interp_indices = self.right_interp_indices
            batched_left_interp_indices = (self.left_interp_indices + batch_offset).view(-1)
        else:
            batched_left_interp_indices = self.left_interp_indices.view(-1)

        flattened_right_interp_right_res = right_interp_right_res.view(batch_size * n_inducing, n_vecs)
        selected_right_vals = flattened_right_interp_right_res.index_select(0, batched_left_interp_indices)
        selected_right_vals = selected_right_vals.view(*batch_shape, n_left_rows, n_left_interp, n_vecs)
        left_values_grad = (selected_right_vals * left_vecs.unsqueeze(-2)).sum(-1)

        # right_interp_values_grad
        left_interp_left_res = self.base_lazy_tensor._t_matmul(left_res).contiguous()
        batch_shape = left_interp_left_res.shape[:-2]
        batch_size = batch_shape.numel()
        if len(batch_shape):
            batch_offset = torch.arange(0, batch_size, dtype=torch.long, device=self.device).view(*batch_shape)
            batch_offset.unsqueeze_(-1).unsqueeze_(-1).mul_(n_inducing)
            batched_right_interp_indices = (self.right_interp_indices + batch_offset).view(-1)
        else:
            batched_right_interp_indices = self.right_interp_indices.view(-1)

        flattened_left_interp_left_res = left_interp_left_res.view(batch_size * n_inducing, n_vecs)
        selected_left_vals = flattened_left_interp_left_res.index_select(0, batched_right_interp_indices)
        selected_left_vals = selected_left_vals.view(*batch_shape, n_right_rows, n_right_interp, n_vecs)
        right_values_grad = (selected_left_vals * right_vecs.unsqueeze(-2)).sum(-1)

        # Return zero grad for interp indices
        res = tuple(
            base_lv_grad
            + [
                torch.zeros_like(self.left_interp_indices),
                left_values_grad,
                torch.zeros_like(self.right_interp_indices),
                right_values_grad,
            ]
        )
        return res

    def _size(self):
        return torch.Size(
            self.base_lazy_tensor.batch_shape + (self.left_interp_indices.size(-2), self.right_interp_indices.size(-2))
        )

    def _transpose_nonbatch(self):
        res = self.__class__(
            self.base_lazy_tensor.transpose(-1, -2),
            self.right_interp_indices,
            self.right_interp_values,
            self.left_interp_indices,
            self.left_interp_values,
            **self._kwargs,
        )
        return res

    def _sparse_left_interp_t(self, left_interp_indices_tensor, left_interp_values_tensor):
        if hasattr(self, "_sparse_left_interp_t_memo"):
            if torch.equal(self._left_interp_indices_memo, left_interp_indices_tensor) and torch.equal(
                self._left_interp_values_memo, left_interp_values_tensor
            ):
                return self._sparse_left_interp_t_memo

        left_interp_t = sparse.make_sparse_from_indices_and_values(
            left_interp_indices_tensor, left_interp_values_tensor, self.base_lazy_tensor.size()[-2]
        )
        self._left_interp_indices_memo = left_interp_indices_tensor
        self._left_interp_values_memo = left_interp_values_tensor
        self._sparse_left_interp_t_memo = left_interp_t
        return self._sparse_left_interp_t_memo

    def _sparse_right_interp_t(self, right_interp_indices_tensor, right_interp_values_tensor):
        if hasattr(self, "_sparse_right_interp_t_memo"):
            if torch.equal(self._right_interp_indices_memo, right_interp_indices_tensor) and torch.equal(
                self._right_interp_values_memo, right_interp_values_tensor
            ):
                return self._sparse_right_interp_t_memo

        right_interp_t = sparse.make_sparse_from_indices_and_values(
            right_interp_indices_tensor, right_interp_values_tensor, self.base_lazy_tensor.size()[-1]
        )
        self._right_interp_indices_memo = right_interp_indices_tensor
        self._right_interp_values_memo = right_interp_values_tensor
        self._sparse_right_interp_t_memo = right_interp_t
        return self._sparse_right_interp_t_memo

    def _sum_batch(self, dim):
        left_interp_indices = self.left_interp_indices
        left_interp_values = self.left_interp_values
        right_interp_indices = self.right_interp_indices
        right_interp_values = self.right_interp_values

        # Increase interpolation indices appropriately
        left_factor = torch.arange(0, left_interp_indices.size(dim), dtype=torch.long, device=self.device)
        left_factor = _pad_with_singletons(left_factor, 0, self.dim() - dim - 1)
        left_factor = left_factor * self.base_lazy_tensor.size(-2)
        left_interp_indices = left_interp_indices.add(left_factor)
        right_factor = torch.arange(0, right_interp_indices.size(dim), dtype=torch.long, device=self.device)
        right_factor = _pad_with_singletons(right_factor, 0, self.dim() - dim - 1)
        right_factor = right_factor * self.base_lazy_tensor.size(-1)
        right_interp_indices = right_interp_indices.add(right_factor)

        # Rearrange the indices and values
        permute_order = (*range(0, dim), *range(dim + 1, self.dim()), dim)
        left_shape = (*left_interp_indices.shape[:dim], *left_interp_indices.shape[dim + 1 : -1], -1)
        right_shape = (*right_interp_indices.shape[:dim], *right_interp_indices.shape[dim + 1 : -1], -1)
        left_interp_indices = left_interp_indices.permute(permute_order).reshape(left_shape)
        left_interp_values = left_interp_values.permute(permute_order).reshape(left_shape)
        right_interp_indices = right_interp_indices.permute(permute_order).reshape(right_shape)
        right_interp_values = right_interp_values.permute(permute_order).reshape(right_shape)

        # Make the base_lazy tensor block diagonal
        from .block_diag_lazy_tensor import BlockDiagLazyTensor

        block_diag = BlockDiagLazyTensor(self.base_lazy_tensor, block_dim=dim)

        # Finally! We have an interpolated lazy tensor again
        return InterpolatedLazyTensor(
            block_diag, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

    def diag(self):
        if isinstance(self.base_lazy_tensor, RootLazyTensor) and isinstance(self.base_lazy_tensor.root, NonLazyTensor):
            left_interp_vals = left_interp(
                self.left_interp_indices, self.left_interp_values, self.base_lazy_tensor.root.evaluate()
            )
            right_interp_vals = left_interp(
                self.right_interp_indices, self.right_interp_values, self.base_lazy_tensor.root.evaluate()
            )
            return (left_interp_vals * right_interp_vals).sum(-1)
        else:
            return super(InterpolatedLazyTensor, self).diag()

    def matmul(self, tensor):
        # We're using a custom matmul here, because it is significantly faster than
        # what we get from the function factory.
        # The _matmul_closure is optimized for repeated calls, such as for inv_matmul

        if tensor.ndimension() == 1:
            is_vector = True
            tensor = tensor.unsqueeze(-1)
        else:
            is_vector = False

        # right_interp^T * tensor
        base_size = self.base_lazy_tensor.size(-1)
        right_interp_res = left_t_interp(self.right_interp_indices, self.right_interp_values, tensor, base_size)

        # base_lazy_tensor * right_interp^T * tensor
        base_res = self.base_lazy_tensor.matmul(right_interp_res)

        # left_interp * base_lazy_tensor * right_interp^T * tensor
        res = left_interp(self.left_interp_indices, self.left_interp_values, base_res)

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    def zero_mean_mvn_samples(self, num_samples):
        base_samples = self.base_lazy_tensor.zero_mean_mvn_samples(num_samples)
        batch_iter = tuple(range(1, base_samples.dim()))
        base_samples = base_samples.permute(*batch_iter, 0)
        res = left_interp(self.left_interp_indices, self.left_interp_values, base_samples).contiguous()
        batch_iter = tuple(range(res.dim() - 1))
        return res.permute(-1, *batch_iter).contiguous()
