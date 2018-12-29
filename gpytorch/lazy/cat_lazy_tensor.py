#!/usr/bin/env python3

import torch
import itertools
from .lazy_tensor import LazyTensor


class CatLazyTensor(LazyTensor):
    """
    A `LazyTensor` that represents the concatenation of `LazyTensor`s.
    Each LazyTensor must have the same shape except in the concatenating
    dimension.

    Args:
        - :attr:`lazy_tensors` (list of LazyTensors):
            A list of LazyTensors whose sizes are the same except in
            concatenating dimension :attr:`dim`.
        - :attr:`dim` (int):
            The concatenating dimension which can be a batch dimension.
    """

    def __init__(self, *lazy_tensors, dim=0, output_device=None):
        if len(lazy_tensors) == 0:
            raise RuntimeError("List of LazyTensors must be non-empty")
        if not all([isinstance(t, LazyTensor) for t in lazy_tensors]):
            raise RuntimeError("CatLazyTensor requires a list of all LazyTensors")
        super(CatLazyTensor, self).__init__(*lazy_tensors, dim=dim,
                                            output_device=output_device)

        def remove_dim(tuple, dim):
            return tuple[:dim] + tuple[dim + 1:]

        rep_tensor = lazy_tensors[0]
        ndims = rep_tensor.ndimension()
        if dim < 0:
            dim = ndims + dim
        pre_cat_size = tuple(rep_tensor.size()[:dim])
        post_cat_size = tuple(rep_tensor.size()[dim + 1:])

        cat_dim_len = 0
        cat_dim_sizes = []
        tensor_idx_to_start_idx = []
        for t_idx, t in enumerate(lazy_tensors):
            if t.ndimension() != ndims:
                raise RuntimeError("All tensors must have the same number of dimensions")
            if remove_dim(t.size(), dim) != remove_dim(rep_tensor.size(), dim):
                raise RuntimeError("All LazyTensors must have the same size in "
                                   "the non-concatenation dimension")
            tensor_idx_to_start_idx.append(cat_dim_len)
            cat_dim_size = t.size()[dim]
            cat_dim_len += cat_dim_size
            cat_dim_sizes.append(cat_dim_size)

        # using itertools to more quickly join list of lists
        idx_to_tensor_idx = [[t_idx] * size for t_idx, size in enumerate(cat_dim_sizes)]
        idx_to_tensor_idx = list(itertools.chain.from_iterable(idx_to_tensor_idx))

        self.lazy_tensors = lazy_tensors
        self.pre_cat_size = pre_cat_size
        self.post_cat_size = post_cat_size
        self.cat_dim_sizes = cat_dim_sizes
        self.cat_dim_len = cat_dim_len
        # can't call this attribute self.dim because LazyTensor has a dim() function
        self.cat_dim = dim
        self.idx_to_tensor_idx = idx_to_tensor_idx
        self.tensor_idx_to_start_idx = tensor_idx_to_start_idx
        self.output_device = output_device

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        indices = list(batch_indices) + [left_indices, right_indices]
        cat_dim_indices = indices[self.cat_dim]

        if torch.is_tensor(cat_dim_indices):
            slices = list(cat_dim_indices.cpu().numpy())
            sliced_tensors = [self.idx_to_tensor_idx[idx] for idx in slices]
        elif isinstance(cat_dim_indices, slice):
            if cat_dim_indices == slice(None, None, None):
                res_list = [t._get_indices(left_indices, right_indices, *batch_indices)
                            for t in self.lazy_tensors]
                return self.__class__(res_list, dim=self.cat_dim)

            slices = list(range(self.cat_dim_len))[cat_dim_indices]
            sliced_tensors = self.idx_to_tensor_idx[cat_dim_indices]

        # map indices to slice object
        d = {}
        start = slices[0]
        stop = start
        last_slice = slices[0]
        last_t_idx = sliced_tensors[0]
        for slice_, t_idx in zip(slices, sliced_tensors):
            step = slice_ - last_slice
            stop += step
            if t_idx != last_t_idx:
                step = None if step == 1 else step
                d[t_idx] = slice(start, stop + 1, step)

                start = slice_
                stop = start
            last_slice, last_t_idx = slice_, t_idx
        step = None if step == 1 else step
        d[t_idx] = slice(start, stop + 1, step)

        if len(d) == 1:
            t_idx = sliced_tensors[0]
            t = self.lazy_tensors[t_idx]
            indices[self.cat_dim] = d[t_idx]
            indices = tuple(indices)
            return t._get_indices(indices[-2], indices[-1], *indices[:-2])

        res_list = []
        for t_idx, t in zip(sliced_tensors, self.lazy_tensors[sliced_tensors]):
            indices[self.cat_dim] = d[t_idx]
            indices = tuple(indices)
            res = t._get_indices(indices[-2], indices[-1], *indices[:-2])
            res_list.append(res)

        return self.__class__(res_list, dim=self.cat_dim)

    def _matmul(self, rhs):
        isvector = rhs.ndimension() == 1
        if isvector:
            rhs = rhs.unsqueeze(1)

        output_device = (self.output_device if self.output_device is not None
                         else rhs.device)
        # make a copy of `rhs` on each device
        rhs_ = []
        for d in self.devices:
            if d != rhs.device:
                rhs_.append(rhs.to(d))
            else:
                rhs_.append(rhs)

        if self.cat_dim == self.ndimension() - 2:
            res_list = [t._matmul(rhs)
                        for t, rhs in zip(self.lazy_tensors, rhs_)]
            # copy result back to output device
            res_list = [x.to(output_device) for x in res_list]
            res = torch.cat(res_list, dim=self.cat_dim)
        elif self.cat_dim == self.ndimension() - 1:
            curr_idx = 0
            res_list = []
            for t, size, rhs in zip(self.lazy_tensors, self.cat_dim_sizes, rhs_):
                res_list.append(t._matmul(rhs[curr_idx:curr_idx + size]))
                curr_idx += size
            # copy result back to output device
            res_list = [x.to(output_device) for x in res_list]
            res = torch.sum(torch.stack(res_list), dim=0)
        else:
            while rhs.ndimension() < self.ndimension():
                rhs = rhs.unsqueeze(0)
            curr_idx = 0
            res_list = []
            index = [slice(None, None, None) for _ in range(self.ndimension())]
            for t, size, rhs in zip(self.lazy_tensors, self.cat_dim_sizes, rhs_):
                index[self.cat_dim] = slice(curr_idx, curr_idx + size, None)
                res_list.append(t._matmul(rhs[index]))
                curr_idx += size
            # copy result back to output device
            res_list = [x.to(output_device) for x in res_list]
            res = torch.cat(res_list, dim=self.cat_dim)

        if isvector:
            res = res.squeeze(-1)
        return res

    def _size(self):
        size = self.pre_cat_size + (self.cat_dim_len,) + self.post_cat_size
        return torch.Size(size)

    def _transpose_nonbatch(self):
        if self.cat_dim == self.ndimension() - 2:
            new_dim = self.cat_dim + 1
        elif self.cat_dim == self.ndimension() - 1:
            new_dim = self.cat_dim - 1
        else:
            new_dim = self.cat_dim
        return self.__class__(*[t._transpose_nonbatch()
                                for t in self.lazy_tensors], dim=new_dim)

    @property
    def devices(self):
        return [t.device for t in self.lazy_tensors]

    @property
    def device_count(self):
        return len(set(self.devices))
