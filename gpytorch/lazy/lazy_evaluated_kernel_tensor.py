#!/usr/bin/env python3

import torch

from .. import settings
from ..utils.memoize import cached
from ..utils.getitem import _noop_index
from .lazy_tensor import LazyTensor
from .lazy_tensor_representation_tree import LazyTensorRepresentationTree
from .non_lazy_tensor import lazify


LAZY_KERNEL_TENSOR_WARNING = (
    "A LazyEvaluatedKernelTensor is not intended to be used directly as a tensor! Call evaluate() first."
)


class LazyEvaluatedKernelTensor(LazyTensor):
    def __init__(self, kernel, x1, x2, batch_dims=None, **params):
        super(LazyEvaluatedKernelTensor, self).__init__(kernel, x1, x2, **params)
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

        return LazyEvaluatedKernelTensor(
            self.kernel, x1, x2, **self.params
        )

    def _matmul(self, rhs):
        raise RuntimeError(LAZY_KERNEL_TENSOR_WARNING)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        raise RuntimeError(LAZY_KERNEL_TENSOR_WARNING)

    def _size(self):
        size = self.kernel.size(self.x1, self.x2)
        if self.batch_dims == (0, 2):
            if len(size) == 2:
                return torch.Size((self.x1.size(-1), size[0], size[1]))
            else:
                return torch.Size((self.x1.size(-1) * size[0], size[1], size[2]))
        return size

    def _transpose_nonbatch(self):
        return self.__class__(self.kernel, self.x2, self.x1, **self.params)

    def _t_matmul(self, rhs):
        raise RuntimeError(LAZY_KERNEL_TENSOR_WARNING)

    def diag(self):
        """
        Getting the diagonal of a kernel can be handled more efficiently by
        transposing the batch and data dimension before calling the kernel.
        Implementing it this way allows us to compute predictions more efficiently
        in cases where only the variances are required.
        """
        from ..kernels import Kernel

        if hasattr(self, "_cached_kernel_diag"):
            return self._cached_kernel_diag
        elif hasattr(self, "_cached_kernel_eval"):
            return self._cached_kernel_eval.diag()
        else:
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
            self._cached_kernel_diag = res.view(self.shape[:-1]).contiguous()
            return self._cached_kernel_diag

    def evaluate_kernel(self):
        """
        NB: This is a meta LazyTensor, in the sense that evaluate can return
        a LazyTensor if the kernel being evaluated does so.
        """
        if hasattr(self, "_cached_kernel_eval"):
            return self._cached_kernel_eval
        else:
            if not self.is_batch:
                x1 = self.x1.unsqueeze(0)
                x2 = self.x2.unsqueeze(0)
            else:
                x1 = self.x1
                x2 = self.x2

            with settings.lazily_evaluate_kernels(False):
                self._cached_kernel_eval = self.kernel(
                    x1, x2, diag=False, batch_dims=self.batch_dims, **self.params
                )

            if (
                not self.is_batch
                and self._cached_kernel_eval.ndimension() == 3
                and self._cached_kernel_eval.size(0) == 1
            ):
                self._cached_kernel_eval = self._cached_kernel_eval[0]

            self._cached_kernel_eval = lazify(self._cached_kernel_eval)
            return self._cached_kernel_eval

    @cached
    def evaluate(self):
        return self.evaluate_kernel().evaluate()

    def repeat(self, *sizes):
        if len(sizes) == 3:
            x1 = self.x1.repeat(sizes[0], sizes[1], 1)
            x2 = self.x2.repeat(sizes[0], sizes[1], 1)
        elif len(sizes) == 2 and x1.ndim() == 2:
            x1 = self.x1.repeat(sizes[0], 1)
            x2 = self.x2.repeat(sizes[0], 1)
        else:
            raise RuntimeError("Invalid number of sizes (expected 2 or 3)")

        return LazyEvaluatedKernelTensor(self.kernel, x1, x2, **self.params)

    def representation(self):
        return self.evaluate_kernel().representation()

    def representation_tree(self):
        return LazyTensorRepresentationTree(self.evaluate_kernel())
