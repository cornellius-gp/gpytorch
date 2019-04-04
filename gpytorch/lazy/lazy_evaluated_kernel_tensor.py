#!/usr/bin/env python3

import torch

from .. import settings, beta_features
from ..utils.memoize import cached
from ..utils.getitem import _noop_index
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify


class LazyEvaluatedKernelTensor(LazyTensor):
    _check_size = False

    def _check_args(self, x1, x2, kernel, batch_dims=None, **params):
        if not torch.is_tensor(x1):
            return "x1 must be a tensor. Got {}".format(x1.__class__.__name__)
        if not torch.is_tensor(x2):
            return "x1 must be a tensor. Got {}".format(x1.__class__.__name__)

    def __init__(self, x1, x2, kernel, batch_dims=None, **params):
        super(LazyEvaluatedKernelTensor, self).__init__(
            x1, x2, kernel=kernel, batch_dims=batch_dims, **params
        )
        self.kernel = kernel
        self.x1 = x1
        self.x2 = x2
        self.batch_dims = batch_dims
        self.params = params

    @property
    def dtype(self):
        return self.kernel.dtype

    @property
    def device(self):
        return self.x1.device

    def _expand_batch(self, batch_shape):
        return self.evaluate_kernel()._expand_batch(batch_shape)

    def _getitem(self, row_index, col_index, *batch_indices):
        x1 = self.x1
        if self.batch_dims == (0, 2):
            x1 = x1.permute(0, 2, 1).contiguous()
            x1 = x1.view(-1, x1.size(-1), 1)
        try:
            x1 = x1[(*batch_indices, row_index, _noop_index)]
        # We're going to handle multi-batch indexing with a try-catch loop
        # This way - in the default case, we can avoid doing expansions of x1 which can be timely
        except IndexError:
            if isinstance(batch_indices, slice):
                x1 = x1.expand(1, *self.x1.shape[-2:])[(*batch_indices, row_index, _noop_index)]
            elif isinstance(batch_indices, tuple):
                if any([not isinstance(bi, slice) for bi in batch_indices]):
                    raise RuntimeError(
                        f"Attempting to tensor index a non-batch matrix's batch dimensions. "
                        "Got batch index {batch_indices} but my shape was {self.shape}"
                    )
                x1 = x1.expand(
                    *([1] * len(batch_indices)),
                    *self.x1.shape[-2:]
                )[(*batch_indices, row_index, _noop_index)]
        x2 = self.x2
        if self.batch_dims == (0, 2):
            x2 = x2.permute(0, 2, 1).contiguous()
            x2 = x2.view(-1, x2.size(-1), 1)
        try:
            x2 = x2[(*batch_indices, col_index, _noop_index)]
        # We're going to handle multi-batch indexing with a try-catch loop
        # This way - in the default case, we can avoid doing expansions of x1 which can be timely
        except IndexError:
            if isinstance(batch_indices, slice):
                x2 = x2.expand(1, *self.x2.shape[-2:])[(*batch_indices, row_index, _noop_index)]
            elif isinstance(batch_indices, tuple):
                if any([not isinstance(bi, slice) for bi in batch_indices]):
                    raise RuntimeError(
                        f"Attempting to tensor index a non-batch matrix's batch dimensions. "
                        "Got batch index {batch_indices} but my shape was {self.shape}"
                    )
                x2 = x2.expand(
                    *([1] * len(batch_indices)),
                    *self.x2.shape[-2:]
                )[(*batch_indices, row_index, _noop_index)]

        return self.__class__(
            x1, x2, kernel=self.kernel, **self.params
        )

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
                    self.kernel(sub_x1, x2, diag=False, batch_dims=self.batch_dims, **self.params)
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

        x1 = self.x1.detach()
        x2 = self.x2.detach()

        # Break objects into chunks
        sub_x1s = torch.split(x1, split_size, dim=-2)
        sub_left_vecss = torch.split(left_vecs, split_size, dim=-2)
        # Compute the gradient in chunks
        for sub_x1, sub_left_vecs in zip(sub_x1s, sub_left_vecss):
            with torch.enable_grad(), settings.lazily_evaluate_kernels(False):
                sub_kernel_matrix = lazify(
                    self.kernel(sub_x1, x2, diag=False, batch_dims=self.batch_dims, **self.params)
                )
            sub_grad_outputs = tuple(sub_kernel_matrix._quad_form_derivative(sub_left_vecs, right_vecs))
            sub_kernel_outputs = tuple(sub_kernel_matrix.representation())
            torch.autograd.backward(sub_kernel_outputs, sub_grad_outputs)

        return x1.grad, x2.grad

    @cached(name="size")
    def _size(self):
        size = self.kernel.size(self.x1, self.x2)
        if self.batch_dims == (0, 2):
            return torch.Size((self.x1.size(-1), ) + size)
        return size

    def _transpose_nonbatch(self):
        return self.__class__(
            self.x2, self.x1, kernel=self.kernel, batch_dims=self.batch_dims, **self.params
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

        res = super(Kernel, self.kernel).__call__(x1, x2, diag=True, batch_dims=self.batch_dims, **self.params)

        # Now we'll make sure that the shape we're getting from diag makes sense
        if settings.debug.on():
            # If we used batch_dims...
            shape = self.kernel.size(x1, x2)
            if self.batch_dims == (0, 2):
                expected_shape = torch.Size((x1.size(-1),) + shape[:-1])
                if res.shape != expected_shape:
                    raise RuntimeError(
                        "The kernel {} is not equipped to handle batch_dims=(0, 2) "
                        "and diag. Expected size {}. Got size {}.".format(
                            self.__class__.__name__, expected_shape, res.shape
                        )
                    )

            # If we didn't use batch_dims...
            else:
                expected_shape = shape[:-1]
                if res.shape != expected_shape:
                    raise RuntimeError(
                        "The kernel {} is not equipped to handle and diag. Expected size {}. "
                        "Got size {}".format(self.__class__.__name__, expected_shape, res.shape)
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
            res = self.kernel(
                x1, x2, diag=False, batch_dims=self.batch_dims, **self.params
            )
            self.kernel.active_dims = temp_active_dims

        return lazify(res)

    @cached
    def evaluate(self):
        return self.evaluate_kernel().evaluate()

    def ndimension(self):
        # TODO: fix this with the remove_batch_dim PR
        return max(self.x2.dim(), self.x2.dim())

    def repeat(self, *repeats):
        if len(repeats) == 1 and hasattr(repeats[0], "__iter__"):
            repeats = repeats[0]
        *batch_repeat, row_repeat, col_repeat = repeats

        x1 = self.x1.repeat(*batch_repeat, row_repeat, 1)
        x2 = self.x2.repeat(*batch_repeat, col_repeat, 1)
        return LazyEvaluatedKernelTensor(self.kernel, x1, x2, **self.params)

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
        if index[0] is Ellipsis and all([isinstance(idx, slice) for idx in index[1:]]):
            *batch_indices, row_index, col_index = index
            return self._getitem(row_index, col_index, *batch_indices)
        else:
            return super().__getitem__(index)
