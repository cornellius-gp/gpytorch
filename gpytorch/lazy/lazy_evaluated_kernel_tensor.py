#!/usr/bin/env python3

import torch

from .. import beta_features, settings
from ..utils import broadcasting
from ..utils.getitem import _noop_index
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify


class LazyEvaluatedKernelTensor(LazyTensor):
    _check_size = False

    def _check_args(self, x1, x2, kernel, last_dim_is_batch=False, **params):
        if not torch.is_tensor(x1):
            return "x1 must be a tensor. Got {}".format(x1.__class__.__name__)
        if not torch.is_tensor(x2):
            return "x1 must be a tensor. Got {}".format(x1.__class__.__name__)

    def __init__(self, x1, x2, kernel, last_dim_is_batch=False, **params):
        super(LazyEvaluatedKernelTensor, self).__init__(
            x1, x2, kernel=kernel, last_dim_is_batch=last_dim_is_batch, **params
        )
        self.kernel = kernel
        self.x1 = x1
        self.x2 = x2
        self.last_dim_is_batch = last_dim_is_batch
        self.params = params

    @property
    def dtype(self):
        return self.kernel.dtype

    @property
    def device(self):
        return self.x1.device

    @property
    def requires_grad(self):
        return super().requires_grad or any(param.requires_grad for param in self.kernel.parameters())

    def _set_requires_grad(self, val):
        super()._set_requires_grad(val)
        # The behavior that differs from the base LazyTensor setter
        for param in self.kernel.parameters():
            param.requires_grad_(val)

    def _expand_batch(self, batch_shape):
        return self.evaluate_kernel()._expand_batch(batch_shape)

    def _getitem(self, row_index, col_index, *batch_indices):
        x1 = self.x1
        x2 = self.x2
        num_outs_per_in = self.kernel.num_outputs_per_input(x1, x2)

        # The row index and col index should exactly correspond to which entries of x1 and x2 we need
        # So we'll basically call x1[*batch_indices, row_index, :], x2[*batch_indices, col_index, :]

        # However - if we have multiple outputs per input, then the indices won't directly
        # correspond to the entries of row/col. We'll have to do a little pre-processing
        if num_outs_per_in != 1:
            if not isinstance(x1, slice) or not isinstance(x2, slice):
                # It's too complicated to deal with tensor indices in this case - we'll use the super method
                return self.evaluate_kernel()._getitem(row_index, col_index, *batch_indices)

            # Now we know that x1 and x2 are slices
            # Let's make sure that the slice dimensions perfectly correspond with the number of
            # outputs per input that we have
            row_start, row_end, row_step = row_index.start, row_index.stop, row_index.step
            col_start, col_end, col_step = col_index.start, col_index.stop, col_index.step
            if row_step is not None or col_step is not None:
                return self.evaluate_kernel()._getitem(row_index, col_index, *batch_indices)
            if (
                (row_start % num_outs_per_in)
                or (col_start % num_outs_per_in)
                or (row_end % num_outs_per_in)
                or (col_end % num_outs_per_in)
            ):
                return self.evaluate_kernel()._getitem(row_index, col_index, *batch_indices)

            # Otherwise - let's divide the slices by the number of outputs per input
            row_index = slice(row_start // num_outs_per_in, row_end // num_outs_per_in, None)
            col_index = slice(col_start // num_outs_per_in, col_end // num_outs_per_in, None)

        # Define the index we're using for the last index
        # If the last index corresponds to a batch, then we'll use the appropriate batch_index
        # Otherwise, we'll use the _noop_index
        if self.last_dim_is_batch:
            *batch_indices, dim_index = batch_indices
        else:
            dim_index = _noop_index

        # Get the indices of x1 and x2 that matter for the kernel
        # Call x1[*batch_indices, row_index, :]
        try:
            x1 = x1[(*batch_indices, row_index, dim_index)]
        # We're going to handle multi-batch indexing with a try-catch loop
        # This way - in the default case, we can avoid doing expansions of x1 which can be timely
        except IndexError:
            if any(not isinstance(bi, slice) for bi in batch_indices):
                raise RuntimeError(
                    "Attempting to tensor index a non-batch matrix's batch dimensions. "
                    f"Got batch index {batch_indices} but my shape was {self.shape}"
                )
            x1 = x1.expand(*([1] * (len(batch_indices) - self.x1.dim() + 2)), *self.x1.shape)
            x1 = x1[(*batch_indices, row_index, dim_index)]

        # Call x2[*batch_indices, col_index, :]
        try:
            x2 = x2[(*batch_indices, col_index, dim_index)]
        # We're going to handle multi-batch indexing with a try-catch loop
        # This way - in the default case, we can avoid doing expansions of x1 which can be timely
        except IndexError:
            if any([not isinstance(bi, slice) for bi in batch_indices]):
                raise RuntimeError(
                    "Attempting to tensor index a non-batch matrix's batch dimensions. "
                    f"Got batch index {batch_indices} but my shape was {self.shape}"
                )
            x2 = x2.expand(*([1] * (len(batch_indices) - self.x1.dim() + 2)), *self.x2.shape)
            x2 = x2[(*batch_indices, col_index, dim_index)]

        if len(batch_indices) == 0 or all(ind == slice(None, None, None) for ind in batch_indices):
            new_kernel = self.kernel  # Avoid unnecessary copying when we aren't explicitly indexing batch dims
        else:
            new_kernel = self.kernel.__getitem__(batch_indices)

        # Now construct a kernel with those indices
        return self.__class__(x1, x2, kernel=new_kernel, last_dim_is_batch=self.last_dim_is_batch, **self.params)

    def _matmul(self, rhs):
        # This _matmul is defined computes the kernel in chunks
        # It is only used when we are using kernel checkpointing
        # It won't be called if checkpointing is off
        x1 = self.x1
        x2 = self.x2

        split_size = beta_features.checkpoint_kernel.value()
        if not split_size:
            raise RuntimeError(
                "Should not have ended up in LazyEvaluatedKernelTensor._matmul without kernel checkpointing. "
                "This is probably a bug in GPyTorch."
            )

        with torch.no_grad(), settings.lazily_evaluate_kernels(False):
            sub_x1s = torch.split(x1, split_size, dim=-2)
            res = []
            for sub_x1 in sub_x1s:
                sub_kernel_matrix = lazify(
                    self.kernel(sub_x1, x2, diag=False, last_dim_is_batch=self.last_dim_is_batch, **self.params)
                )
                res.append(sub_kernel_matrix._matmul(rhs))

            res = torch.cat(res, dim=-2)
            return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        # This _quad_form_derivative computes the kernel in chunks
        # It is only used when we are using kernel checkpointing
        # It won't be called if checkpointing is off
        split_size = beta_features.checkpoint_kernel.value()
        if not split_size:
            raise RuntimeError(
                "Should not have ended up in LazyEvaluatedKernelTensor._quad_form_derivative without kernel "
                "checkpointing. This is probably a bug in GPyTorch."
            )

        x1 = self.x1.detach().requires_grad_(True)
        x2 = self.x2.detach().requires_grad_(True)

        # Break objects into chunks
        sub_x1s = [sub_x1.detach() for sub_x1 in torch.split(x1, split_size, dim=-2)]
        sub_left_vecss = torch.split(left_vecs, split_size, dim=-2)
        # Compute the gradient in chunks
        for sub_x1, sub_left_vecs in zip(sub_x1s, sub_left_vecss):
            sub_x1.requires_grad_(True)
            with torch.enable_grad(), settings.lazily_evaluate_kernels(False):
                sub_kernel_matrix = lazify(
                    self.kernel(sub_x1, x2, diag=False, last_dim_is_batch=self.last_dim_is_batch, **self.params)
                )
            sub_grad_outputs = tuple(sub_kernel_matrix._quad_form_derivative(sub_left_vecs, right_vecs))
            sub_kernel_outputs = tuple(sub_kernel_matrix.representation())
            torch.autograd.backward(sub_kernel_outputs, sub_grad_outputs)

        x1.grad = torch.cat([sub_x1.grad.data for sub_x1 in sub_x1s], dim=-2)
        return x1.grad, x2.grad

    @cached(name="size")
    def _size(self):
        if settings.debug.on():
            if hasattr(self.kernel, "size"):
                raise RuntimeError("Kernels must define `num_outputs_per_input` and should not define `size`")

        x1 = self.x1
        x2 = self.x2
        num_outputs_per_input = self.kernel.num_outputs_per_input(x1, x2)
        num_rows = x1.size(-2) * num_outputs_per_input
        num_cols = x2.size(-2) * num_outputs_per_input

        # Default case - when we're not using broadcasting
        # We write this case special for efficiency
        if x1.shape[:-2] == x2.shape[:-2] and x1.shape[:-2] == self.kernel.batch_shape:
            expected_size = self.kernel.batch_shape + torch.Size((num_rows, num_cols))

        # When we're using broadcasting
        else:
            expected_size = broadcasting._matmul_broadcast_shape(
                torch.Size([*x1.shape[:-2], num_rows, x1.size(-1)]),
                torch.Size([*x2.shape[:-2], x2.size(-1), num_cols]),
                error_msg="x1 and x2 were not broadcastable to a proper kernel shape. "
                "Got x1.shape = {} and x2.shape = {}".format(str(x1.shape), str(x2.shape)),
            )
            expected_size = (
                broadcasting._mul_broadcast_shape(
                    expected_size[:-2],
                    self.kernel.batch_shape,
                    error_msg=(
                        f"x1 and x2 were not broadcastable with kernel of batch_shape {self.kernel.batch_shape}. "
                        f"Got x1.shape = {x1.shape} and x2.shape = {x2.shape}"
                    ),
                )
                + expected_size[-2:]
            )

        # Handle when the last dim is batch
        if self.last_dim_is_batch:
            expected_size = expected_size[:-2] + x1.shape[-1:] + expected_size[-2:]
        return expected_size

    def _transpose_nonbatch(self):
        return self.__class__(
            self.x2, self.x1, kernel=self.kernel, last_dim_is_batch=self.last_dim_is_batch, **self.params
        )

    def add_jitter(self, jitter_val=1e-3):
        return self.evaluate_kernel().add_jitter(jitter_val)

    def _unsqueeze_batch(self, dim):
        return self.evaluate_kernel()._unsqueeze_batch(dim)

    @cached(name="kernel_diag")
    def diag(self):
        """
        Getting the diagonal of a kernel can be handled more efficiently by
        transposing the batch and data dimension before calling the kernel.
        Implementing it this way allows us to compute predictions more efficiently
        in cases where only the variances are required.
        """
        from ..kernels import Kernel

        x1 = self.x1
        x2 = self.x2

        res = super(Kernel, self.kernel).__call__(
            x1, x2, diag=True, last_dim_is_batch=self.last_dim_is_batch, **self.params
        )

        # Now we'll make sure that the shape we're getting from diag makes sense
        if settings.debug.on():
            expected_shape = self.shape[:-1]
            if res.shape != expected_shape:
                raise RuntimeError(
                    "The kernel {} is not equipped to handle and diag. Expected size {}. "
                    "Got size {}".format(self.kernel.__class__.__name__, expected_shape, res.shape)
                )

        if isinstance(res, LazyTensor):
            res = res.evaluate()
        return res.view(self.shape[:-1]).contiguous()

    @cached(name="kernel_eval")
    def evaluate_kernel(self):
        """
        NB: This is a meta LazyTensor, in the sense that evaluate can return
        a LazyTensor if the kernel being evaluated does so.
        """
        x1 = self.x1
        x2 = self.x2

        with settings.lazily_evaluate_kernels(False):
            temp_active_dims = self.kernel.active_dims
            self.kernel.active_dims = None
            res = self.kernel(x1, x2, diag=False, last_dim_is_batch=self.last_dim_is_batch, **self.params)
            self.kernel.active_dims = temp_active_dims

        # Check the size of the output
        if settings.debug.on():
            if res.shape != self.shape:
                raise RuntimeError(
                    f"The expected shape of the kernel was {self.shape}, but got {res.shape}. "
                    "This is likely a bug in GPyTorch."
                )

        return lazify(res)

    @cached
    def evaluate(self):
        return self.evaluate_kernel().evaluate()

    def repeat(self, *repeats):
        if len(repeats) == 1 and hasattr(repeats[0], "__iter__"):
            repeats = repeats[0]
        *batch_repeat, row_repeat, col_repeat = repeats

        x1 = self.x1.repeat(*batch_repeat, row_repeat, 1)
        x2 = self.x2.repeat(*batch_repeat, col_repeat, 1)
        return self.__class__(x1, x2, kernel=self.kernel, last_dim_is_batch=self.last_dim_is_batch, **self.params)

    def representation(self):
        # If we're checkpointing the kernel, we'll use chunked _matmuls defined in LazyEvaluatedKernelTensor
        if beta_features.checkpoint_kernel.value():
            return super().representation()
        # Otherwise, we'll evaluate the kernel (or at least its LazyTensor representation) and use its
        # representation
        else:
            return self.evaluate_kernel().representation()

    def representation_tree(self):
        # If we're checkpointing the kernel, we'll use chunked _matmuls defined in LazyEvaluatedKernelTensor
        if beta_features.checkpoint_kernel.value():
            return super().representation_tree()
        # Otherwise, we'll evaluate the kernel (or at least its LazyTensor representation) and use its
        # representation
        else:
            return self.evaluate_kernel().representation_tree()

    def __getitem__(self, index):
        """
        Supports subindexing of the matrix this LazyTensor represents. This may return either another
        :obj:`gpytorch.lazy.LazyTensor` or a :obj:`torch.tensor` depending on the exact implementation.
        """
        # Process the index
        index = index if isinstance(index, tuple) else (index,)
        # Special case for the most common case: [..., slice, slice]
        if len(index) == 3 and index[0] is Ellipsis and isinstance(index[1], slice) and isinstance(index[2], slice):
            _, row_index, col_index = index
            batch_indices = [slice(None, None, None)] * (self.dim() - 2)
            return self._getitem(row_index, col_index, *batch_indices)
        else:
            return super().__getitem__(index)
