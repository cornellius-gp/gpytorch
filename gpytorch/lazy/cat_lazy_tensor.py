#!/usr/bin/env python3

import torch

from .. import settings
from ..utils.broadcasting import _matmul_broadcast_shape, _mul_broadcast_shape
from ..utils.deprecation import bool_compat
from ..utils.getitem import _noop_index
from .lazy_tensor import LazyTensor, delazify
from .non_lazy_tensor import NonLazyTensor, lazify


def cat(inputs, dim=0, output_device=None):
    if all(torch.is_tensor(i) for i in inputs):
        return torch.cat(inputs, dim=dim)

    inputs = [lazify(i) for i in inputs]

    if all(isinstance(i, NonLazyTensor) for i in inputs):
        # Dont form a CatLazyTensor if all tensors are NonLazyTensor
        return lazify(torch.cat([delazify(i) for i in inputs], dim=dim))

    if output_device is None and all(i.device == inputs[0].device for i in inputs):
        output_device = inputs[0].device
    elif output_device is None:
        raise RuntimeError("Trying to concat lazy tensors on different devices without specifying an output device.")

    return CatLazyTensor(*inputs, dim=dim, output_device=output_device)


class CatLazyTensor(LazyTensor):
    r"""
    A `LazyTensor` that represents the concatenation of other lazy tensors.
    Each LazyTensor must have the same shape except in the concatenating
    dimension.

    Args:
        - :attr:`lazy_tensors` (list of LazyTensors):
            A list of LazyTensors whose sizes are the same except in
            concatenating dimension :attr:`dim`
        - :attr:`dim` (int):
            The concatenating dimension which can be a batch dimension.
        - :attr:`output_device` (torch.device):
            The CatLazyTensor will appear to appear on :attr:`output_device`
            and place any output `torch.Tensors` on :attr:`output_device`
    """

    def _check_args(self, *lazy_tensors, dim=0, output_device=None):
        if len(lazy_tensors) == 0:
            raise RuntimeError("List of LazyTensors must be non-empty")
        elif len(lazy_tensors) == 1:
            raise RuntimeError("Why are we trying to concatenate a single LazyTensor?")
        if not all([isinstance(t, LazyTensor) for t in lazy_tensors]):
            raise RuntimeError("CatLazyTensor requires a list of all LazyTensors")

        rep_tensor = lazy_tensors[0]
        rep_tensor_noncat_shape = list(rep_tensor.shape)
        del rep_tensor_noncat_shape[dim]

        for t in lazy_tensors:
            if t.dim() != rep_tensor.dim():
                raise RuntimeError("All tensors must have the same number of dimensions")

            t_noncat_shape = list(t.shape)
            del t_noncat_shape[dim]
            if t_noncat_shape != rep_tensor_noncat_shape:
                raise RuntimeError("All LazyTensors must have the same size in " "the non-concatenation dimension")

    def __init__(self, *lazy_tensors, dim=0, output_device=None):
        # Make sure index is negative index
        rep_tensor = lazy_tensors[0]
        ndims = rep_tensor.ndimension()
        if dim >= 0:
            positive_dim = dim
            dim = dim - ndims
        else:
            positive_dim = ndims + dim

        # Standard initialization
        super().__init__(*lazy_tensors, dim=dim, output_device=output_device)
        self.lazy_tensors = lazy_tensors
        self.cat_dim = dim
        self.output_device = output_device

        # Helpers for _getitem
        cat_dim_sizes = torch.tensor([t.size(dim) for t in lazy_tensors], device=output_device)
        cat_dim_cum_sizes = torch.zeros(len(lazy_tensors) + 1, dtype=torch.long, device=output_device)
        torch.cumsum(cat_dim_sizes, dim=-1, out=cat_dim_cum_sizes[1:])
        idx_to_tensor_idx = torch.empty(cat_dim_cum_sizes[-1].item(), dtype=torch.long, device=output_device)
        for tsr_idx, (start_idx, end_idx) in enumerate(zip(cat_dim_cum_sizes[:-1], cat_dim_cum_sizes[1:])):
            idx_to_tensor_idx[start_idx.item() : end_idx.item()].fill_(tsr_idx)

        self.cat_dim_sizes = cat_dim_sizes
        self.cat_dim_cum_sizes = cat_dim_cum_sizes
        self.idx_to_tensor_idx = idx_to_tensor_idx
        self._shape = torch.Size(
            (*rep_tensor.shape[:positive_dim], cat_dim_cum_sizes[-1].item(), *rep_tensor.shape[positive_dim + 1 :])
        )

    def _split_slice(self, slice_idx):
        """
        Splits a slice(a, b, None) in to a list of slices [slice(a1, b1, None), slice(a2, b2, None), ...]
        so that each slice in the list slices in to a single tensor that we have concatenated with this LazyTensor.
        """
        if slice_idx.step is not None:
            # TODO: Add support for this eventually.
            raise RuntimeError("Slicing a CatLazyTensor with a step is not currently supported!")
        start_idx = slice_idx.start if slice_idx.start is not None else 0
        stop_idx = slice_idx.stop if slice_idx.stop is not None else self.size(self.cat_dim)

        first_tensor_idx = self.idx_to_tensor_idx[start_idx].item()
        last_tensor_idx = self.idx_to_tensor_idx[stop_idx - 1].item()

        first_tensor_start_index = start_idx - self.cat_dim_cum_sizes[first_tensor_idx].item()
        last_tensor_stop_index = stop_idx - self.cat_dim_cum_sizes[last_tensor_idx].item()

        if first_tensor_idx == last_tensor_idx:
            return [first_tensor_idx], [slice(first_tensor_start_index, last_tensor_stop_index, None)]
        else:
            num_middle_tensors = last_tensor_idx - first_tensor_idx - 1
            first_slice = slice(first_tensor_start_index, None, None)
            last_slice = slice(None, last_tensor_stop_index, None)
            return (
                list(range(first_tensor_idx, last_tensor_idx + 1)),
                [first_slice] + [_noop_index] * num_middle_tensors + [last_slice],
            )

    def _expand_batch(self, batch_shape):
        batch_dim = self.cat_dim + 2
        if batch_dim < 0:
            if batch_shape[batch_dim] != self.batch_shape[batch_dim]:
                raise RuntimeError(
                    f"Trying to expand a CatLazyTensor in dimension {self.cat_dim}, but this is the concatenated "
                    f"dimension.\nCurrent shape: {self.shape} - expanded shape: {batch_shape + self.matrix_shape}."
                )
            lazy_tensors = []
            for lazy_tensor in self.lazy_tensors:
                sub_batch_shape = list(batch_shape).copy()
                sub_batch_shape[batch_dim] = lazy_tensor.shape[self.cat_dim]
                lazy_tensors.append(lazy_tensor._expand_batch(sub_batch_shape))
        else:
            lazy_tensors = [lazy_tensor._expand_batch(batch_shape) for lazy_tensor in self.lazy_tensors]
        res = self.__class__(*lazy_tensors, dim=self.cat_dim, output_device=self.output_device)
        return res

    def _get_indices(self, row_index, col_index, *batch_indices):
        indices = [*batch_indices, row_index, col_index]
        target_shape = _mul_broadcast_shape(*[index.shape for index in indices])
        indices = [index.expand(target_shape).reshape(-1) for index in indices]
        cat_dim_indices = indices[self.cat_dim]

        # Find out for which indices we switch to different tensors
        target_tensors = self.idx_to_tensor_idx[cat_dim_indices]
        does_switch_tensor = torch.ones(target_tensors.numel() + 1, dtype=bool_compat, device=self.device)
        torch.ne(target_tensors[:-1], target_tensors[1:], out=does_switch_tensor[1:-1])

        # Get the LazyTensors that will comprise the new LazyTensor
        lazy_tensor_indices = target_tensors[does_switch_tensor[:-1]].tolist()
        lazy_tensors = [self.lazy_tensors[idx] for idx in lazy_tensor_indices]

        # Get the new set of indices for each of the LazyTensors
        switch_tensor = does_switch_tensor.nonzero(as_tuple=False).squeeze(-1)
        split_sizes = (switch_tensor[1:] - switch_tensor[:-1]).tolist()
        sub_indices = zip(
            *[
                list(index.split(split_sizes)) if torch.is_tensor(index) else [index] * len(split_sizes)
                for index in indices
            ]
        )
        # Make everything a list
        sub_indices = [list(sub_index) for sub_index in sub_indices]

        # Make sure that we have adjusted the start and ends of the indices that correspond to the cat dim
        for lazy_tensor_idx, sub_index in zip(lazy_tensor_indices, sub_indices):
            sub_index[self.cat_dim] = sub_index[self.cat_dim] - self.cat_dim_cum_sizes[lazy_tensor_idx]

        res_list = [
            lazy_tensor._get_indices(sub_index[-2], sub_index[-1], *sub_index[:-2])
            for lazy_tensor, sub_index in zip(lazy_tensors, sub_indices)
        ]
        if len(res_list) == 1:
            return res_list[0].view(target_shape).to(self.device)
        else:
            return torch.cat(res_list).view(target_shape).to(self.device)

    def _getitem(self, row_index, col_index, *batch_indices):
        indices = [*batch_indices, row_index, col_index]
        cat_dim_indices = indices[self.cat_dim]

        if isinstance(cat_dim_indices, slice):
            if cat_dim_indices == _noop_index:
                res_list = [
                    lazy_tensor._getitem(row_index, col_index, *batch_indices) for lazy_tensor in self.lazy_tensors
                ]

            else:
                res_list = []
                tensor_idxs, target_slices = self._split_slice(cat_dim_indices)
                for tensor_idx, target_slice in zip(tensor_idxs, target_slices):
                    indices[self.cat_dim] = target_slice
                    res = self.lazy_tensors[tensor_idx]._getitem(indices[-2], indices[-1], *indices[:-2])
                    res_list.append(res)

        elif torch.is_tensor(cat_dim_indices):
            # Find out for which indices we switch to different tensors
            target_tensors = self.idx_to_tensor_idx[cat_dim_indices]
            does_switch_tensor = torch.ones(target_tensors.numel() + 1, dtype=bool_compat, device=self.device)
            torch.ne(target_tensors[:-1], target_tensors[1:], out=does_switch_tensor[1:-1])

            # Get the LazyTensors that will comprise the new LazyTensor
            lazy_tensor_indices = target_tensors[does_switch_tensor[:-1]].tolist()
            lazy_tensors = [self.lazy_tensors[idx] for idx in lazy_tensor_indices]

            # Get the new set of indices for each of the LazyTensors
            switch_tensor = does_switch_tensor.nonzero(as_tuple=False).squeeze(-1)
            split_sizes = (switch_tensor[1:] - switch_tensor[:-1]).tolist()
            sub_indices = zip(
                *[
                    list(index.split(split_sizes)) if torch.is_tensor(index) else [index] * len(split_sizes)
                    for index in indices
                ]
            )
            # Make everything a list
            sub_indices = [list(sub_index) for sub_index in sub_indices]

            # Make sure that we have adjusted the start and ends of the indices that correspond to the cat dim
            for lazy_tensor_idx, sub_index in zip(lazy_tensor_indices, sub_indices):
                sub_index[self.cat_dim] = sub_index[self.cat_dim] - self.cat_dim_cum_sizes[lazy_tensor_idx]

            res_list = [
                lazy_tensor._getitem(sub_index[-2], sub_index[-1], *sub_index[:-2])
                for lazy_tensor, sub_index in zip(lazy_tensors, sub_indices)
            ]

        elif isinstance(cat_dim_indices, int):  # Should only happen for cat on batch dim
            target_tensor = self.idx_to_tensor_idx[cat_dim_indices].item()
            cat_dim_indices = cat_dim_indices - self.cat_dim_cum_sizes[target_tensor]
            indices[self.cat_dim] = cat_dim_indices
            res_list = [self.lazy_tensors[target_tensor]._getitem(indices[-2], indices[-1], *indices[:-2])]

        # Process the list
        if len(res_list) == 1:
            return res_list[0].to(self.output_device)
        else:
            res = self.__class__(*res_list, dim=self.cat_dim, output_device=self.output_device)
            return res

    def _matmul(self, rhs):
        output_device = self.device if self.device is not None else rhs.device
        # make a copy of `rhs` on each device
        rhs_ = []
        for d in self.devices:
            if d != rhs.device:
                rhs_.append(rhs.to(d))
            else:
                rhs_.append(rhs)

        if self.cat_dim == -2:
            res_list = [t._matmul(rhs) for t, rhs in zip(self.lazy_tensors, rhs_)]
            # copy result back to output device
            res_list = [x.to(output_device) for x in res_list]
            res = torch.cat(res_list, dim=-2)
        elif self.cat_dim == -1:
            curr_idx = 0
            res_list = []
            index = [slice(None, None, None) for _ in range(rhs.ndimension())]
            for t, size, rhs in zip(self.lazy_tensors, self.cat_dim_sizes, rhs_):
                index[-2] = slice(curr_idx, curr_idx + size, None)
                res_list.append(t._matmul(rhs[index]))
                curr_idx += size
            # copy result back to output device and sum
            res_list = [x.to(output_device) for x in res_list]
            res = 0.0
            for x in res_list:
                res = res + x
        else:
            output_shape = _matmul_broadcast_shape(self.shape, rhs.shape)
            rhs = rhs.expand(*output_shape[:-2], *rhs.shape[-2:])
            curr_idx = 0
            res_list = []
            for t, size in zip(self.lazy_tensors, self.cat_dim_sizes):
                sub_rhs = rhs.narrow(self.cat_dim, curr_idx, size)
                res_list.append(t._matmul(sub_rhs))
                curr_idx += size
            # copy result back to output device
            res_list = [x.to(output_device) for x in res_list]
            res = torch.cat(res_list, dim=self.cat_dim)

        return res

    def _permute_batch(self, *dims):
        lazy_tensors = [lazy_tensor._permute_batch(*dims) for lazy_tensor in self.lazy_tensors]
        if self.cat_dim < -2:
            positive_cat_dim = self.dim() + self.cat_dim
            new_cat_dim = dims.index(positive_cat_dim)
        else:
            new_cat_dim = self.cat_dim
        return self.__class__(*lazy_tensors, dim=new_cat_dim, output_device=self.output_device)

    def _size(self):
        return self._shape

    def _transpose_nonbatch(self):
        if self.cat_dim == -2:
            new_dim = -1
        elif self.cat_dim == -1:
            new_dim = -2
        else:
            new_dim = self.cat_dim
        return self.__class__(
            *[t._transpose_nonbatch() for t in self.lazy_tensors], dim=new_dim, output_device=self.output_device
        )

    def _unsqueeze_batch(self, dim):
        cat_dim = self.dim() + self.cat_dim
        lazy_tensors = [lazy_tensor._unsqueeze_batch(dim) for lazy_tensor in self.lazy_tensors]
        res = self.__class__(
            *lazy_tensors, dim=(cat_dim + 1 if dim <= cat_dim else cat_dim), output_device=self.output_device
        )
        return res

    def diag(self):
        if settings.debug.on():
            if not self.is_square:
                raise RuntimeError("Diag works on square matrices (or batches)")

        if self.cat_dim == -2:
            res = []
            curr_col = 0
            for t in self.lazy_tensors:
                n_rows, n_cols = t.shape[-2:]
                rows = torch.arange(0, n_rows, dtype=torch.long, device=t.device)
                cols = torch.arange(curr_col, curr_col + n_rows, dtype=torch.long, device=t.device)
                res.append(t[..., rows, cols].to(self.device))
                curr_col += n_rows
            res = torch.cat(res, dim=-1)
        elif self.cat_dim == -1:
            res = []
            curr_row = 0
            for t in self.lazy_tensors:
                n_rows, n_cols = t.shape[-2:]
                rows = torch.arange(curr_row, curr_row + n_cols, dtype=torch.long, device=t.device)
                cols = torch.arange(0, n_cols, dtype=torch.long, device=t.device)
                curr_row += n_cols
                res.append(t[..., rows, cols].to(self.device))
            res = torch.cat(res, dim=-1)
        else:
            res = torch.cat([t.diag().to(self.device) for t in self.lazy_tensors], dim=self.cat_dim + 1)
        return res

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        res = super().inv_quad_logdet(inv_quad_rhs, logdet, reduce_inv_quad)
        return tuple(r.to(self.device) for r in res)

    @property
    def device(self):
        return self.output_device

    @property
    def devices(self):
        return [t.device for t in self.lazy_tensors]

    @property
    def device_count(self):
        return len(set(self.devices))

    def to(self, device_id):
        """
        returns a new CatLazyTensor with device_id as the output_device
        Warning: this does not move the LazyTensors in this CatLazyTensor to
        device_id
        """
        new_kwargs = dict(self._kwargs)
        new_kwargs["output_device"] = device_id
        return self.__class__(*self._args, **new_kwargs)

    def all_to(self, device_id):
        """
        Create a new CatLazyTensor with all LazyTensors in CatLazyTensor moved
        to one device device. The new CatLazyTensor also has device_id as the
        output_device.
        """
        new_args = []
        new_kwargs = {}
        for arg in self._args:
            if hasattr(arg, "to"):
                new_args.append(arg.to(device_id))
            else:
                new_args.append(arg)
        for name, val in self._kwargs.items():
            if hasattr(val, "to"):
                new_kwargs[name] = val.to(device_id)
            else:
                new_kwargs[name] = val
        new_kwargs["output_device"] = device_id
        return self.__class__(*new_args, **new_kwargs)
