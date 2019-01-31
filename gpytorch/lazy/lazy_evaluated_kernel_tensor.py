#!/usr/bin/env python3

import torch

from .. import settings, beta_features
from ..utils.memoize import cached
from ..utils.getitem import _noop_index
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify


class LazyEvaluatedKernelTensor(LazyTensor):
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
        self.is_batch = self.x1.ndimension() == 3
        self.params = params

    @property
    def dtype(self):
        return self.x1.dtype

    @property
    def device(self):
        return self.x1.device

    def _expand_batch(self, batch_shape):
        return LazyEvaluatedKernelTensor(
            self.kernel,
            self.x1.expand(*batch_shape, self.x1.shape[-2:]),
            self.x2.expand(*batch_shape, self.x2.shape[-2:]),
            **self.params
        )

    def _getitem(self, row_col_are_absorbed, row_index, col_index, *batch_indices):
        x1 = self.x1
        if self.batch_dims == (0, 2):
            x1 = x1.permute(0, 2, 1).contiguous()
            x1 = x1.view(-1, x1.size(-1), 1)
        x1 = x1[(*batch_indices, row_index, _noop_index)]
        x2 = self.x2
        if self.batch_dims == (0, 2):
            x2 = x2.permute(0, 2, 1).contiguous()
            x2 = x2.view(-1, x2.size(-1), 1)
        x2 = x2[(*batch_indices, col_index, _noop_index)]

        return self.__class__(
            x1, x2, kernel=self.kernel, **self.params
        )

    def _matmul(self, rhs):
        # This _matmul is defined computes the kernel in chunks
        # It is only used when we are using kernel checkpointing
        # It won't be called if checkpointing is off
        if not self.is_batch:
            x1 = self.x1.unsqueeze(0)
            x2 = self.x2.unsqueeze(0)
            rhs = rhs.unsqueeze(0)
        else:
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
            if not self.is_batch:
                res = res.squeeze(0)
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

        if not self.is_batch:
            x1 = self.x1.unsqueeze(0)
            x2 = self.x2.unsqueeze(0)
            left_vecs = left_vecs.unsqueeze(0)
            right_vecs = right_vecs.unsqueeze(0)
        else:
            x1 = self.x1
            x2 = self.x2

        x1 = x1.detach()
        x2 = x2.detach()

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

    def _size(self):
        size = self.kernel.size(self.x1, self.x2)
        if self.batch_dims == (0, 2):
            if len(size) == 2:
                return torch.Size((self.x1.size(-1), size[0], size[1]))
            else:
                return torch.Size((self.x1.size(-1) * size[0], size[1], size[2]))
        return size

    def _transpose_nonbatch(self):
        return self.__class__(
            self.x2, self.x1, kernel=self.kernel, batch_dims=self.batch_dims,
            squeeze_row=self.squeeze_col, squeeze_col=self.squeeze_row, **self.params
        )

    def add_jitter(self, jitter_val=1e-3):
        return self.evaluate_kernel().add_jitter(jitter_val)

    @cached(name="kernel_diag")
    def diag(self):
        """
        Getting the diagonal of a kernel can be handled more efficiently by
        transposing the batch and data dimension before calling the kernel.
        Implementing it this way allows us to compute predictions more efficiently
        in cases where only the variances are required.
        """
        from ..kernels import Kernel

        if not self.is_batch:
            x1 = self.x1.unsqueeze(0)
            x2 = self.x2.unsqueeze(0)
        else:
            x1 = self.x1
            x2 = self.x2

        # If x1 or x2 only has one data point, make sure to unsqueeze the data-size dimension
        if x1.dim() == 2:  # We only have a single data point
            x1 = x1.unsqueeze(1)
        if x2.dim() == 2:  # We only have a single data point
            x2 = x2.unsqueeze(1)

        res = super(Kernel, self.kernel).__call__(x1, x2, diag=True, batch_dims=self.batch_dims, **self.params)

        # Did this Kernel eat the diag option?
        # If it does not return a LazyEvaluatedKernelTensor, we can call diag on the output
        if not isinstance(res, LazyEvaluatedKernelTensor):
            if res.dim() == x1.dim() and res.shape[-2:] == torch.Size((x1.size(-2), x2.size(-2))):
                res = res.diag()

        # Now we'll make sure that the shape we're getting from diag makes sense
        if settings.debug.on():
            # If we used batch_dims...
            shape = self.kernel.size(x1, x2)
            if self.batch_dims == (0, 2):
                if len(shape) == 2:
                    expected_shape = torch.Size((x1.size(-1), shape[0]))
                else:
                    expected_shape = torch.Size((shape[0] * x1.size(-1), shape[1]))
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
        if not self.is_batch:
            x1 = self.x1.unsqueeze(0)
            x2 = self.x2.unsqueeze(0)
        else:
            x1 = self.x1
            x2 = self.x2

        with settings.lazily_evaluate_kernels(False):
            temp_active_dims = self.kernel.active_dims
            self.kernel.active_dims = None
            res = self.kernel(
                x1, x2, diag=False, batch_dims=self.batch_dims, **self.params
            )
            self.kernel.active_dims = temp_active_dims

        if (
            not self.is_batch
            and res.ndimension() == 3
            and res.size(0) == 1
        ):
            res = res[0]

        return lazify(res)

    @cached
    def evaluate(self):
        return self.evaluate_kernel().evaluate()

    def mul(self, other):
        if isinstance(other, LazyEvaluatedKernelTensor):
            other = other.evaluate_kernel()
        return self.evaluate_kernel().mul(other)

    def repeat(self, *sizes):
        x1 = self.x1.repeat(sizes[:-2], sizes[-2], 1)
        x2 = self.x1.repeat(sizes[:-2], sizes[-1], 1)

        return self.__class__(x1, x2, kernel=self.kernel, batch_dims=self.batch_dims, **self.params)

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
