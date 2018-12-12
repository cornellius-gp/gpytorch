#!/usr/bin/env python3

import torch

from ..utils.memoize import cached
from .lazy_tensor import LazyTensor


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
        res_mat_shape = res.shape[len(self.base_lazy_tensor.batch_shape) :]
        constant = self.constant.view(*self.constant.shape, *[1 for i in res_mat_shape])
        return res * constant

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        res = self.base_lazy_tensor._get_indices(left_indices, right_indices, *batch_indices)
        constant = self.constant.__getitem__(batch_indices)
        return res * constant

    def _getitem(self, *indices):
        # NOTE TO FUTURE SELF:
        # This custom __getitem__ is actually very important!
        # It prevents constructing an InterpolatedLazyTensor when one isn't needed
        # This effects runntimes by up to 5x on simple exat GPs
        # Run __getitem__ on the base_lazy_tensor and the constant
        base_lazy_tensor = self.base_lazy_tensor._getitem(*indices)
        constant = self.constant[indices[:-2]]

        if torch.is_tensor(base_lazy_tensor):
            constant = constant.view(*constant.shape, *[1] * (base_lazy_tensor.dim() - constant.dim()))

        return base_lazy_tensor * constant

    def _matmul(self, rhs):
        res = self.base_lazy_tensor._matmul(rhs)
        res_mat_shape = res.shape[len(self.base_lazy_tensor.batch_shape) :]
        constant = self.constant.view(*self.constant.shape, *[1 for i in res_mat_shape])
        res = res * constant
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        # Gradient with respect to the constant
        constant_deriv = left_vecs * self.base_lazy_tensor._matmul(right_vecs)
        constant_deriv = constant_deriv.sum(-2).sum(-1)

        # Get derivaties of everything else
        res_mat_shape = left_vecs.shape[len(self.base_lazy_tensor.batch_shape) :]
        constant = self.constant.view(*self.constant.shape, *[1 for i in res_mat_shape])
        left_vecs = left_vecs * constant
        res = self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs)

        return res + (constant_deriv,)

    def _size(self):
        return self.base_lazy_tensor.size()

    def _t_matmul(self, rhs):
        res = self.base_lazy_tensor._t_matmul(rhs)
        res_mat_shape = res.shape[len(self.base_lazy_tensor.batch_shape) :]
        constant = self.constant.view(*self.constant.shape, *[1 for i in res_mat_shape])
        res = res * constant
        return res

    def _transpose_nonbatch(self):
        return ConstantMulLazyTensor(self.base_lazy_tensor._transpose_nonbatch(), self.constant)

    @property
    def constant(self):
        # Make sure that the constant can be expanded to the appropriate size
        try:
            constant = self._constant.expand(self.base_lazy_tensor.batch_shape)
        except RuntimeError:
            raise RuntimeError(
                "ConstantMulLazyTensor of size {} received an invalid constant of size {}.".format(
                    self.base_lazy_tensor.shape, self._constant.shape
                )
            )

        return constant

    def diag(self):
        res = self.base_lazy_tensor.diag()
        res_mat_shape = res.shape[len(self.base_lazy_tensor.batch_shape) :]
        constant = self.constant.view(*self.constant.shape, *[1 for i in res_mat_shape])
        return res * constant

    @cached
    def evaluate(self):
        res = self.base_lazy_tensor.evaluate()
        res_mat_shape = res.shape[len(self.base_lazy_tensor.batch_shape) :]
        constant = self.constant.view(*self.constant.shape, *[1 for i in res_mat_shape])
        return constant * res
