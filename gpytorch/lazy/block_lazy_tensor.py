#!/usr/bin/env python3

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
        - :attr:`base_lazy_tensor` (LazyTensor):
            A `k x n x n` LazyTensor, or a `bk x n x n` LazyTensor, representing `k` blocks.
        - :attr:`num_blocks` (int or None):
            Set this to `k` for `bk x n x n` batched LazyTensors, or `None` for `k x n x n`
            unbatched LazyTensors.
    """

    def __init__(self, base_lazy_tensor, num_blocks=None):
        if base_lazy_tensor.ndimension() != 3:
            raise RuntimeError("base_lazy_tensor must be a batch matrix (i.e. 3 dimensions)")
        super(BlockLazyTensor, self).__init__(base_lazy_tensor, num_blocks=num_blocks)
        self.base_lazy_tensor = base_lazy_tensor
        self.num_blocks = num_blocks

    def _getitem(self, *indices):
        if self.num_blocks is None:
            res = super(BlockLazyTensor, self)._getitem(*indices)
            return res

        # Cases for when there's an inner batch
        else:
            batch_index = indices[0]
            first_tensor_index_dim = None

            # Keeping all batch dimensions - recursion base case
            if isinstance(batch_index, slice) and batch_index == slice(None, None, None):
                res = super(BlockLazyTensor, self)._getitem(*indices)
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

            # Keep sum_batch_index, because we still have an inner batch
            # Also keep track that there has been tensor indexing
            elif torch.is_tensor(batch_index):
                block_index = torch.arange(0, self.num_blocks, dtype=torch.long, device=self.device)
                batch_index = (batch_index.unsqueeze(1).mul(self.num_blocks) + block_index.unsqueeze(0)).view(-1)
                num_blocks = self.num_blocks
                first_tensor_index_dim = 0

            else:
                raise RuntimeError("Unknown batch index type")

            # Now construct a new sum batch lazy tensor, and recurse
            components = tuple(component[batch_index] for component in self._args)
            new_var = self.__class__(*components, num_blocks=num_blocks)

            # If the index was only on the batch index, we're done
            if len(indices) == 1:
                return new_var

            # Else - recurse
            else:
                left_index = indices[1]
                right_index = indices[2] if len(indices) >= 3 else slice(None, None, None)

                # Normal case if we're indexing the LT with ints or slices
                # Also squeeze dimensions if we're indexing with tensors
                squeeze_left = False
                squeeze_right = False
                if isinstance(left_index, int):
                    left_index = slice(left_index, left_index + 1, None)
                    squeeze_left = True
                elif torch.is_tensor(left_index):
                    squeeze_left = True
                if isinstance(right_index, int):
                    right_index = slice(right_index, right_index + 1, None)
                    squeeze_right = True
                elif torch.is_tensor(right_index):
                    squeeze_right = True

                if torch.is_tensor(left_index) and torch.is_tensor(right_index):
                    if left_index.numel() != right_index.numel():
                        raise RuntimeError(
                            "Expected the tensor indices to be the same size: got {} and {}".format(
                                left_index.numel(), right_index.numel()
                            )
                        )

                    if new_var.ndimension() == 2:
                        return new_var._get_indices(left_index, right_index)

                    else:
                        batch_index = torch.arange(0, new_var.size(0), dtype=torch.long, device=self.device)
                        if first_tensor_index_dim is not None:
                            if batch_index.numel() != left_index.numel():
                                raise RuntimeError(
                                    "Expected the tensor indices to be the same size: got {}, {} and {}".format(
                                        batch_index.numel(), left_index.numel(), right_index.numel()
                                    )
                                )
                            return new_var._get_indices(left_index, right_index, batch_index)
                        else:
                            batch_size = batch_index.numel()
                            row_col_size = left_index.numel()
                            batch_index = batch_index.unsqueeze(1).repeat(1, row_col_size).view(-1)
                            left_index = left_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
                            right_index = right_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
                            res = new_var._get_indices(left_index, right_index, batch_index)
                            return res.view(batch_size, row_col_size)

                # Normal case: we have to do some processing on eithe rthe rows or columns
                res = new_var._getitem_nonbatch(left_index, right_index, first_tensor_index_dim)
                if (squeeze_left or squeeze_right) and isinstance(res, LazyTensor):
                    res = res.evaluate()
                if squeeze_left:
                    res = res.squeeze(-2)
                if squeeze_right:
                    res = res.squeeze(-1)

                return res

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
