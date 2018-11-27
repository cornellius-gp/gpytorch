#!/usr/bin/env python3

import torch
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
        if torch.is_tensor(constant):
            if constant.ndimension() > 1:
                raise RuntimeError(
                    "Got a constant with %d dimensions - expected a 0D or 1D tensor" % constant.ndimension()
                )
            elif constant.numel() > 1:
                if not (base_lazy_tensor.ndimension() == 3 and base_lazy_tensor.size(0) == constant.numel()):
                    numel = constant.numel()
                    raise RuntimeError(
                        "A constant with size %d expedts a 3D lazy var. with batch size %d. "
                        "Got a %dD lazy var. with size %s"
                        % (numel, numel, base_lazy_tensor.ndimension(), repr(base_lazy_tensor.size()))
                    )

            elif constant.numel() == 1:
                constant = constant.squeeze()
        else:
            constant = torch.tensor(constant, device=base_lazy_tensor.device, dtype=base_lazy_tensor.dtype)

        super(ConstantMulLazyTensor, self).__init__(base_lazy_tensor, constant)
        self.base_lazy_tensor = base_lazy_tensor
        self.constant = constant

    def _matmul(self, rhs):
        res = self.base_lazy_tensor._matmul(rhs)
        res = res * self._constant_as(res)
        return res

    def _t_matmul(self, rhs):
        res = self.base_lazy_tensor._t_matmul(rhs)
        res = res * self._constant_as(res)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        res = list(self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs))
        for i, item in enumerate(res):
            if torch.is_tensor(item) and res[i].sum():
                res[i] = res[i] * self._constant_as(res[i])
        # Gradient with respect to the constant
        if self.constant.numel() == 1:
            constant_deriv = (left_vecs * self.base_lazy_tensor._matmul(right_vecs)).sum().expand_as(self.constant)
        else:
            constant_deriv = left_vecs * self.base_lazy_tensor._matmul(right_vecs)
            constant_deriv = constant_deriv.sum(-2, keepdim=True).sum(-1, keepdim=True)

        if constant_deriv.dim():
            constant_deriv = constant_deriv.view(*self.constant.size())
        res.append(constant_deriv)
        return res

    def _constant_as(self, other):
        size = [self.constant.numel()] + [1] * (other.ndimension() - 1)
        constant = self.constant.view(*size)

        if constant.ndimension() > other.ndimension():
            constant = constant.squeeze(-1)

        return constant

    def _size(self):
        return self.base_lazy_tensor.size()

    def _transpose_nonbatch(self):
        return ConstantMulLazyTensor(self.base_lazy_tensor._transpose_nonbatch(), self.constant)

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        res = self.base_lazy_tensor._get_indices(left_indices, right_indices, *batch_indices)
        return self.constant.expand_as(res) * res

    def _approx_diag(self):
        res = self.base_lazy_tensor._approx_diag()
        return res * self._constant_as(res)

    def evaluate(self):
        res = self.base_lazy_tensor.evaluate()
        return res * self._constant_as(res)

    def diag(self):
        res = self.base_lazy_tensor.diag()
        res = res * self._constant_as(res)
        return res

    def __getitem__(self, i):
        constant = self.constant
        if constant.numel() > 1:
            first_index = i[0] if isinstance(i, tuple) else i
            constant = constant[first_index]
        base_lazy_tensor = self.base_lazy_tensor.__getitem__(i)
        if torch.is_tensor(base_lazy_tensor) and constant.dim() < base_lazy_tensor.dim():
            constant = constant.view(constant.numel(), *([1] * (base_lazy_tensor.dim() - 1)))
        return base_lazy_tensor * constant
