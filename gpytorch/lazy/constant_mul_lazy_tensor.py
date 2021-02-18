#!/usr/bin/env python3
from typing import Optional

import torch

from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .root_lazy_tensor import RootLazyTensor


class ConstantMulLazyTensor(LazyTensor):
    """
    A LazyTensor that multiplies a base LazyTensor by a scalar constant:

    ```
    constant_mul_lazy_tensor = constant * base_lazy_tensor
    ```

    .. note::

        To element-wise multiply two lazy tensors, see :class:`gpytorch.lazy.MulLazyTensor`

    Args:
        base_lazy_tensor (LazyTensor) or (b x n x m)): The base_lazy tensor
        constant (Tensor): The constant

    If `base_lazy_tensor` represents a matrix (non-batch), then `constant` must be a
    0D tensor, or a 1D tensor with one element.

    If `base_lazy_tensor` represents a batch of matrices (b x m x n), then `constant` can be
    either:
    - A 0D tensor - the same constant is applied to all matrices in the batch
    - A 1D tensor with one element - the same constant is applied to all matrices
    - A 1D tensor with `b` elements - a different constant is applied to each matrix

    Example::

        >>> base_base_lazy_tensor = gpytorch.lazy.ToeplitzLazyTensor([1, 2, 3])
        >>> constant = torch.tensor(1.2)
        >>> new_base_lazy_tensor = gpytorch.lazy.ConstantMulLazyTensor(base_base_lazy_tensor, constant)
        >>> new_base_lazy_tensor.evaluate()
        >>> # Returns:
        >>> # [[ 1.2, 2.4, 3.6 ]
        >>> #  [ 2.4, 1.2, 2.4 ]
        >>> #  [ 3.6, 2.4, 1.2 ]]
        >>>
        >>> base_base_lazy_tensor = gpytorch.lazy.ToeplitzLazyTensor([[1, 2, 3], [2, 3, 4]])
        >>> constant = torch.tensor([1.2, 0.5])
        >>> new_base_lazy_tensor = gpytorch.lazy.ConstantMulLazyTensor(base_base_lazy_tensor, constant)
        >>> new_base_lazy_tensor.evaluate()
        >>> # Returns:
        >>> # [[[ 1.2, 2.4, 3.6 ]
        >>> #   [ 2.4, 1.2, 2.4 ]
        >>> #   [ 3.6, 2.4, 1.2 ]]
        >>> #  [[ 1, 1.5, 2 ]
        >>> #   [ 1.5, 1, 1.5 ]
        >>> #   [ 2, 1.5, 1 ]]]
    """

    def __init__(self, base_lazy_tensor, constant):
        if not torch.is_tensor(constant):
            constant = torch.tensor(constant, device=base_lazy_tensor.device, dtype=base_lazy_tensor.dtype)

        super(ConstantMulLazyTensor, self).__init__(base_lazy_tensor, constant)
        self.base_lazy_tensor = base_lazy_tensor
        self._constant = constant

    def _approx_diag(self):
        res = self.base_lazy_tensor._approx_diag()
        return res * self._constant.unsqueeze(-1)

    def _expand_batch(self, batch_shape):
        return self.__class__(
            self.base_lazy_tensor._expand_batch(batch_shape),
            self._constant.expand(*batch_shape) if len(batch_shape) else self._constant,
        )

    def _get_indices(self, row_index, col_index, *batch_indices):
        # NOTE TO FUTURE SELF:
        # This custom __getitem__ is actually very important!
        # It prevents constructing an InterpolatedLazyTensor when one isn't needed
        # This affects runntimes by up to 5x on simple exact GPs
        # Run __getitem__ on the base_lazy_tensor and the constant
        base_lazy_tensor = self.base_lazy_tensor._get_indices(row_index, col_index, *batch_indices)
        constant = self._constant.expand(self.batch_shape)[batch_indices]
        return base_lazy_tensor * constant

    def _getitem(self, row_index, col_index, *batch_indices):
        # NOTE TO FUTURE SELF:
        # This custom __getitem__ is actually very important!
        # It prevents constructing an InterpolatedLazyTensor when one isn't needed
        # This affects runntimes by up to 5x on simple exact GPs
        # Run __getitem__ on the base_lazy_tensor and the constant
        base_lazy_tensor = self.base_lazy_tensor._getitem(row_index, col_index, *batch_indices)
        constant = self._constant.expand(self.batch_shape)[batch_indices]
        constant = constant.view(*constant.shape, 1, 1)
        return base_lazy_tensor * constant

    def _matmul(self, rhs):
        res = self.base_lazy_tensor._matmul(rhs)
        res = res * self.expanded_constant
        return res

    def _permute_batch(self, *dims):
        return self.__class__(
            self.base_lazy_tensor._permute_batch(*dims), self._constant.expand(self.batch_shape).permute(*dims)
        )

    def _quad_form_derivative(self, left_vecs, right_vecs):
        # Gradient with respect to the constant
        constant_deriv = left_vecs * self.base_lazy_tensor._matmul(right_vecs)
        constant_deriv = constant_deriv.sum(-2).sum(-1)
        while constant_deriv.dim() > self._constant.dim():
            constant_deriv = constant_deriv.sum(0)
        for i in range(self._constant.dim()):
            if self._constant.size(i) == 1:
                constant_deriv = constant_deriv.sum(i, keepdim=True)

        # Get derivaties of everything else
        left_vecs = left_vecs * self.expanded_constant
        res = self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs)

        return tuple(res) + (constant_deriv,)

    def _size(self):
        return self.base_lazy_tensor.size()

    def _t_matmul(self, rhs):
        res = self.base_lazy_tensor._t_matmul(rhs)
        res = res * self.expanded_constant
        return res

    def _transpose_nonbatch(self):
        return ConstantMulLazyTensor(self.base_lazy_tensor._transpose_nonbatch(), self._constant)

    @property
    def expanded_constant(self):
        # Make sure that the constant can be expanded to the appropriate size
        try:
            constant = self._constant.view(*self._constant.shape, 1, 1)
        except RuntimeError:
            raise RuntimeError(
                "ConstantMulLazyTensor of size {} received an invalid constant of size {}.".format(
                    self.base_lazy_tensor.shape, self._constant.shape
                )
            )

        return constant

    def diag(self):
        res = self.base_lazy_tensor.diag()
        return res * self._constant.unsqueeze(-1)

    @cached
    def evaluate(self):
        res = self.base_lazy_tensor.evaluate()
        return res * self.expanded_constant

    @cached(name="root_decomposition")
    def root_decomposition(self, method: Optional[str] = None):
        if torch.all(self._constant >= 0):
            base_root = self.base_lazy_tensor.root_decomposition(method=method).root
            return RootLazyTensor(ConstantMulLazyTensor(base_root, self._constant ** 0.5))

        return super().root_decomposition(method=method)
