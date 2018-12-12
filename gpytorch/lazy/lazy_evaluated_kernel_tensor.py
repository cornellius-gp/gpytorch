#!/usr/bin/env python3

import torch

from .. import settings
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .lazy_tensor_representation_tree import LazyTensorRepresentationTree
from .non_lazy_tensor import NonLazyTensor


LAZY_KERNEL_TENSOR_WARNING = (
    "A LazyEvaluatedKernelTensor is not intended to be used directly as a tensor! Call evaluate() first."
)


class LazyEvaluatedKernelTensor(LazyTensor):
    def __init__(self, kernel, x1, x2, batch_dims=None, squeeze_row=False, squeeze_col=False, **params):
        super(LazyEvaluatedKernelTensor, self).__init__(kernel, x1, x2, **params)
        self.kernel = kernel
        self.x1 = x1
        self.x2 = x2
        self.batch_dims = batch_dims
        self.squeeze_row = squeeze_row
        self.squeeze_col = squeeze_col
        self.is_batch = self.x1.ndimension() == 3 or (self.x1.ndimension() == 2 and self.squeeze_row)
        self.params = params

    @property
    def dtype(self):
        return self.x1.dtype

    @property
    def device(self):
        return self.x1.device

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        from ..kernels import Kernel

        x1 = self.x1[(*batch_indices, left_indices)].unsqueeze(0)
        x2 = self.x2[(*batch_indices, right_indices)].unsqueeze(0)
        res = super(Kernel, self.kernel).__call__(x1.transpose(0, 1), x2.transpose(0, 1))
        if isinstance(res, LazyTensor):
            res = res.evaluate()
        res = res.view(-1)
        return res

    def _getitem(self, *indices):
        if self.is_batch:
            batch_index = indices[0]
            left_index = indices[1]
            right_index = indices[2]
            squeeze_row = self.squeeze_row
            squeeze_col = self.squeeze_col

            x1 = self.x1
            if self.batch_dims == (0, 2):
                x1 = x1.permute(0, 2, 1).contiguous()
                x1 = x1.view(-1, x1.size(-1), 1)
            x1 = x1[batch_index, left_index, :]
            if x1.dim() == 2 and not isinstance(batch_index, int):
                x1 = x1.unsqueeze(1)
                squeeze_row = True
            x2 = self.x2
            if self.batch_dims == (0, 2):
                x2 = x2.permute(0, 2, 1).contiguous()
                x2 = x2.view(-1, x2.size(-1), 1)
            x2 = x2[batch_index, right_index, :]
            if x2.dim() == 2 and not isinstance(batch_index, int):
                x2 = x2.unsqueeze(1)
                squeeze_col = True

            return LazyEvaluatedKernelTensor(
                self.kernel, x1, x2, squeeze_row=squeeze_row, squeeze_col=squeeze_col, **self.params
            )
        else:
            left_index = indices[0]
            right_index = indices[1]
            squeeze_row = self.squeeze_row
            squeeze_col = self.squeeze_col

            x1 = self.x1[left_index, :]
            if x1.dim() == 1:
                x1 = x1.unsqueeze(1)
                squeeze_row = True
            x2 = self.x2[right_index, :]
            if x2.dim() == 1:
                x2 = x2.unsqueeze(1)
                squeeze_col = True

            return LazyEvaluatedKernelTensor(
                self.kernel, x1, x2, squeeze_row=squeeze_row, squeeze_col=squeeze_col, **self.params
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
        from ..kernels import Kernel

        if hasattr(self, "_cached_kernel_eval"):
            return self._cached_kernel_eval
        else:
            if not self.is_batch:
                x1 = self.x1.unsqueeze(0)
                x2 = self.x2.unsqueeze(0)
            else:
                x1 = self.x1
                x2 = self.x2

            self._cached_kernel_eval = super(Kernel, self.kernel).__call__(
                x1, x2, diag=False, batch_dims=self.batch_dims, **self.params
            )
            if self.squeeze_row:
                self._cached_kernel_eval.squeeze_(-2)
            if self.squeeze_col:
                self._cached_kernel_eval.squeeze_(-1)

            if (
                not self.is_batch
                and self._cached_kernel_eval.ndimension() == 3
                and self._cached_kernel_eval.size(0) == 1
            ):
                self._cached_kernel_eval = self._cached_kernel_eval[0]
            if not isinstance(self._cached_kernel_eval, LazyTensor):
                self._cached_kernel_eval = NonLazyTensor(self._cached_kernel_eval)

            return self._cached_kernel_eval

    @cached
    def evaluate(self):
        return self.evaluate_kernel().evaluate()

    def repeat(self, *sizes):
        if self.squeeze_row or self.squeeze_col:
            raise RuntimeError("Can't repeat a row/col of a LazyEvaluatedKernelTensor")
        elif len(sizes) == 3:
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
