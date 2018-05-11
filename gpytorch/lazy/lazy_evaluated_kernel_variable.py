from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .lazy_variable import LazyVariable
from .non_lazy_variable import NonLazyVariable
from .lazy_variable_representation_tree import LazyVariableRepresentationTree
import torch


LAZY_KERNEL_TENSOR_WARNING = (
    "A LazyEvaluatedKernelVariable is not intended to be used directly " "as a tensor! Call evaluate() first."
)


class LazyEvaluatedKernelVariable(LazyVariable):
    def __init__(self, kernel, x1, x2, **params):
        super(LazyEvaluatedKernelVariable, self).__init__(kernel, x1, x2, **params)
        self.kernel = kernel
        self.x1 = x1
        self.x2 = x2
        self.is_batch = self.x1.ndimension() == 3

    def _matmul(self, rhs):
        raise RuntimeError(LAZY_KERNEL_TENSOR_WARNING)

    def _t_matmul(self, rhs):
        raise RuntimeError(LAZY_KERNEL_TENSOR_WARNING)

    def _quad_form_derivative(self, left_vecs, right_vecs):
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
            res = super(Kernel, self.kernel).__call__(x1.transpose(-2, -3), x2.transpose(-2, -3))
            if isinstance(res, LazyVariable):
                res = res.evaluate()
            self._cached_kernel_diag = res.transpose(-3, -2).squeeze()
            return self._cached_kernel_diag

    def evaluate_kernel(self):
        """
        NB: This is a meta LazyVariable, in the sense that evaluate can return
        a LazyVariable if the kernel being evaluated does so.
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
            self._cached_kernel_eval = super(Kernel, self.kernel).__call__(x1, x2)

            if not self.is_batch:
                self._cached_kernel_eval = self._cached_kernel_eval[0]
            if not isinstance(self._cached_kernel_eval, LazyVariable):
                self._cached_kernel_eval = NonLazyVariable(self._cached_kernel_eval)
            return self._cached_kernel_eval

    def representation(self):
        return self.evaluate_kernel().representation()

    def representation_tree(self):
        return LazyVariableRepresentationTree(self.evaluate_kernel())

    def evaluate(self):
        return self.evaluate_kernel().evaluate()

    def __getitem__(self, index):
        index = list(index) if isinstance(index, tuple) else [index]
        ndimension = self.ndimension()
        index += [slice(None, None, None)] * (ndimension - len(index))
        if self.is_batch:
            batch_index = index[0]
            left_index = index[1]
            right_index = index[2]
            return LazyEvaluatedKernelVariable(
                self.kernel, self.x1[batch_index, left_index, :], self.x2[batch_index, right_index, :]
            )
        else:
            left_index = index[0]
            right_index = index[1]
            return LazyEvaluatedKernelVariable(self.kernel, self.x1[left_index, :], self.x2[right_index, :])

    def _size(self):
        if self.is_batch:
            return torch.Size((self.x1.size(0), self.x1.size(-2), self.x2.size(-2)))
        else:
            return torch.Size((self.x1.size(-2), self.x2.size(-2)))
