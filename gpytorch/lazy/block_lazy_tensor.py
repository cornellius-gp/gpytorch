from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .lazy_tensor import LazyTensor


class BlockLazyTensor(LazyTensor):
    """
    An abstract LazyTensor class for block tensors.
    Super classes will determine how the different blocks are layed out
    (e.g. block diagonal, sum over blocks, etc.)

    BlockLazyTensors represent the groups of blocks as a batched Tensor.
    For example, a `k x n x n` tensor represents `k` `n x n` blocks.

    For a batched block tensor, the batch dimension is used to represent
    the actual ("true") batches as well as the different blocks.
    For example, `k` `b x n x n` blocks would be represented as a `bk x n x n`
    Tensor, where the "outer" batch dimension represents the true batch dimension
    (i.e. - the Tensor could be viewed as a `b x k x n x n` Tensor without re-ordering).

    For batch mode, the :attr:`num_blocks` attribute specifes the number of blocks (to differentiate
    from true batches). This attribute should be `None` for non-batched Tensors.

    Args:
        :attr:`base_lazy_tensor` (LazyTensor):
            A `k x n x n` LazyTensor, or a `bk x n x n` LazyTensor, representing `k` blocks.
        :attr:`num_blocks` (int or None):
            Set this to `k` for `bk x n x n` batched LazyTensors, or `None` for `k x n x n`
            unbatched LazyTensors.
    """
    def __init__(self, base_lazy_tensor, num_blocks=None):
        if base_lazy_tensor.ndimension() != 3:
            raise RuntimeError("base_lazy_tensor must be a batch matrix (i.e. 3 dimensions)")
        super(BlockLazyTensor, self).__init__(base_lazy_tensor, num_blocks=num_blocks)
        self.base_lazy_tensor = base_lazy_tensor
        self.num_blocks = num_blocks

    def _transpose_nonbatch(self):
        return self.__class__(self.base_lazy_tensor._transpose_nonbatch(), num_blocks=self.num_blocks)

    def mul(self, other):
        # We're using a custom method here - the constant mul is applied to the base_lazy tensor
        # This preserves the block structure

        if not (torch.is_tensor(other) or isinstance(other, LazyTensor)) or (
            torch.is_tensor(other) and other.numel() == 1
        ):
            from .constant_mul_lazy_tensor import ConstantMulLazyTensor

            return self.__class__(ConstantMulLazyTensor(self.base_lazy_tensor, other), num_blocks=self.num_blocks)
        else:
            return super(BlockLazyTensor, self).mul(other)

    def __getitem__(self, index):
        if self.num_blocks is None:
            res = super(BlockLazyTensor, self).__getitem__(index)
            return res

        # Cases for when there's an inner batch
        else:
            batch_index = index if isinstance(index, int) else index[0]

            # Keeping all batch dimensions - recursion base case
            if batch_index == slice(None, None, None):
                res = super(BlockLazyTensor, self).__getitem__(index)
                return res

            # Construct a new lazy tensor
            # Get rid of sum_batch_index if we're choosing one batch tensor
            if isinstance(batch_index, int):
                batch_index = slice(batch_index * self.num_blocks, (batch_index + 1) * self.num_blocks, None)
                num_blocks = None

            # Keep sum_batch_index, because we still have an inner batch
            elif isinstance(batch_index, slice):
                start, stop, step = batch_index.indices(self.size(0))
                batch_index = slice(start * self.num_blocks, stop * self.num_blocks, step)
                num_blocks = self.num_blocks

            else:
                raise RuntimeError("Unknown batch index type")

            # Now construct a new sum batch lazy tensor, and recurse
            components = tuple(component[batch_index] for component in self._args)
            new_var = self.__class__(*components, num_blocks=num_blocks)

            # If the index was only on the batch index, we're done
            if isinstance(index, int) or len(index) == 1:
                return new_var

            # Else - recurse
            else:
                if new_var.num_blocks is None:
                    index = index[1:]
                else:
                    index = list(index)
                    index[0] = slice(None, None, None)
                return new_var.__getitem__(tuple(index))
