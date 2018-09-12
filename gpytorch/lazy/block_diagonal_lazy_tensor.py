from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Variable
from .lazy_tensor import LazyTensor


class BlockDiagonalLazyTensor(LazyTensor):
    def __init__(self, base_lazy_tensor, n_blocks=None):
        """
        Represents a lazy tensor that is the block diagonal of square matrices
        This tensor is stored as a batch (i.e. a tensor of batch_size x _ x _)
        Therefore, all the block diagonal components must be the same lazy tensor
        type and size

        By specifying n_blocks, you can have two batch tensors: one
        that is summed over, and one that is not
        (i.e. the input is the representation of a tensor that is
         (true_batch_size * n_blocks x _ x _),
         and will return (true_batch_size x _ x _))
        """
        if base_lazy_tensor.ndimension() != 3:
            raise RuntimeError("Base lazy tensor must be a batch matrix (i.e. 3 dimensions)")
        super(BlockDiagonalLazyTensor, self).__init__(base_lazy_tensor, n_blocks=n_blocks)
        self.base_lazy_tensor = base_lazy_tensor
        self.n_blocks = n_blocks

    def _matmul(self, rhs):
        block_size = self.base_lazy_tensor.size(-1)
        isvector = rhs.ndimension() == 1
        if isvector:
            rhs = rhs.unsqueeze(1)

        n_cols = rhs.size(-1)
        rhs = rhs.contiguous().view(-1, block_size, n_cols)

        res = self.base_lazy_tensor._matmul(rhs)
        if self.n_blocks is not None:
            res = res.contiguous().view(-1, self.n_blocks * res.size(1), res.size(2))
        else:
            res = res.contiguous().view(res.size(0) * res.size(1), res.size(2))

        if isvector:
            res = res.squeeze(-1)
        return res

    def _t_matmul(self, rhs):
        block_size = self.base_lazy_tensor.size(-1)
        isvector = rhs.ndimension() == 1
        if isvector:
            rhs = rhs.unsqueeze(1)

        n_cols = rhs.size(-1)
        rhs = rhs.contiguous().view(-1, block_size, n_cols)

        res = self.base_lazy_tensor._t_matmul(rhs)
        if self.n_blocks is not None:
            res = res.contiguous().view(-1, self.n_blocks * res.size(1), res.size(2))
        else:
            res = res.contiguous().view(res.size(0) * res.size(1), res.size(2))

        if isvector:
            res = res.squeeze(-1)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        block_size = self.base_lazy_tensor.size(-1)
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)
        left_vecs = left_vecs.view(-1, block_size, left_vecs.size(-1))
        right_vecs = right_vecs.view(-1, block_size, right_vecs.size(-1))
        res = self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        return res

    def _size(self):
        base_size = self.base_lazy_tensor.size()
        if self.n_blocks is None:
            return torch.Size((base_size[0] * base_size[1], base_size[0] * base_size[2]))
        else:
            true_batch_size = self.base_lazy_tensor.size(0) // self.n_blocks
            return torch.Size((true_batch_size, self.n_blocks * base_size[1], self.n_blocks * base_size[2]))

    def _transpose_nonbatch(self):
        return BlockDiagonalLazyTensor(self.base_lazy_tensor._transpose_nonbatch())

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        block_size = self.base_lazy_tensor.size(-1)
        left_batch_indices = left_indices.div(block_size).long()
        right_batch_indices = left_indices.div(block_size).long()
        batch_indices = batch_indices * block_size + left_batch_indices
        left_indices = left_indices.fmod(block_size)
        right_indices = left_indices.fmod(block_size)

        res = self.base_lazy_tensor._batch_get_indices(batch_indices, left_indices, right_indices)
        res = res * torch.eq(left_batch_indices, right_batch_indices).type_as(res)
        return res

    def _get_indices(self, left_indices, right_indices):
        block_size = self.base_lazy_tensor.size(-1)
        left_batch_indices = left_indices.div(block_size).long()
        right_batch_indices = left_indices.div(block_size).long()
        left_indices = left_indices.fmod(block_size)
        right_indices = left_indices.fmod(block_size)

        res = self.base_lazy_tensor._batch_get_indices(left_batch_indices, left_indices, right_indices)
        res = res * torch.eq(left_batch_indices, right_batch_indices).type_as(res)
        return res

    def diag(self):
        res = self.base_lazy_tensor.diag().contiguous()
        if self.n_blocks:
            res = res.view(res.size(0) // self.n_blocks, -1)
        else:
            res = res.view(-1)
        return res

    def mul(self, other):
        # We're using a custom method here - the constant mul is applied to the base_lazy tensor
        # This preserves the sum batch structure
        if not (isinstance(other, Variable) or isinstance(other, LazyTensor)) or (
            isinstance(other, Variable) and other.numel() == 1
        ):
            from .constant_mul_lazy_tensor import ConstantMulLazyTensor

            return self.__class__(ConstantMulLazyTensor(self.base_lazy_tensor, other), n_blocks=self.n_blocks)
        else:
            return super(BlockDiagonalLazyTensor, self).mul(other)

    def zero_mean_mvn_samples(self, n_samples):
        res = self.base_lazy_tensor.zero_mean_mvn_samples(n_samples)
        if self.n_blocks is None:
            res = res.view(-1, n_samples)
        else:
            res = res.view(self.base_lazy_tensor.size(0) // self.n_blocks, -1, n_samples)
        return res

    def __getitem__(self, index):
        if self.n_blocks is None:
            return super(BlockDiagonalLazyTensor, self).__getitem__(index)

        # Cases for when there's an inner batch
        else:
            batch_index = index if isinstance(index, int) else index[0]

            # Keeping all batch dimensions - recursion base case
            if batch_index == slice(None, None, None):
                res = super(BlockDiagonalLazyTensor, self).__getitem__(index)
                return res

            # Construct a new lazy tensor
            # Get rid of sum_batch_index if we're choosing one batch tensor
            if isinstance(batch_index, int):
                batch_index = slice(batch_index * self.n_blocks, (batch_index + 1) * self.n_blocks, None)
                n_blocks = None

            # Keep sum_batch_index, because we still have an inner batch
            elif isinstance(batch_index, slice):
                start, stop, step = batch_index.indices(self.size(0))
                batch_index = slice(start * self.n_blocks, stop * self.n_blocks, step)
                n_blocks = self.n_blocks

            else:
                raise RuntimeError("Unknown batch index type")

            # Now construct a new sum batch lazy tensor, and recurse
            components = tuple(component[batch_index] for component in self._args)
            new_var = self.__class__(*components, n_blocks=n_blocks)

            # If the index was only on the batch index, we're done
            if isinstance(index, int) or len(index) == 1:
                return new_var

            # Else - recurse
            else:
                if new_var.n_blocks is None:
                    index = index[1:]
                else:
                    index = list(index)
                    index[0] = slice(None, None, None)
                return new_var.__getitem__(tuple(index))
