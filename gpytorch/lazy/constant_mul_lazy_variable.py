from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .lazy_variable import LazyVariable


class ConstantMulLazyVariable(LazyVariable):
    """
    A LazyVariable that multiplies a base LazyVariable by a scalar constant:

    ```
    constant_mul_lazy_variable = constant * base_lazy_variable
    ```

    .. note::
    To element-wise multiply two lazy variables, see :class:`gpytorch.lazy.MulLazyVariable`

    Args:
        lazy_var (LazyVariable) or (b x n x m)): The base lazy variable
        constant (Tensor): The constant

    If `lazy_var` represents a matrix (non-batch), then `constant` must be a
    0D tensor, or a 1D tensor with one element.

    If `lazy_var` represents a batch of matrices (b x m x n), then `constant` can be
    either:
    - A 0D tensor - the same constant is applied to all matrices in the batch
    - A 1D tensor with one element - the same constant is applied to all matrices
    - A 1D tensor with `b` elements - a different constant is applied to each matrix

    Example::

        >>> base_lazy_var = gpytorch.lazy.ToeplitzLazyVariable([1, 2, 3])
        >>> constant = torch.tensor(1.2)
        >>> new_lazy_var = gpytorch.lazy.ConstantMulLazyVariable(base_lazy_var, constant)
        >>> new_lazy_var.evaluate()
        >>> # Returns:
        >>> # [[ 1.2, 2.4, 3.6 ]
        >>> #  [ 2.4, 1.2, 2.4 ]
        >>> #  [ 3.6, 2.4, 1.2 ]]
        >>>
        >>> base_lazy_var = gpytorch.lazy.ToeplitzLazyVariable([[1, 2, 3], [2, 3, 4]])
        >>> constant = torch.Tensor([1.2, 0.5])
        >>> new_lazy_var = gpytorch.lazy.ConstantMulLazyVariable(base_lazy_var, constant)
        >>> new_lazy_var.evaluate()
        >>> # Returns:
        >>> # [[[ 1.2, 2.4, 3.6 ]
        >>> #   [ 2.4, 1.2, 2.4 ]
        >>> #   [ 3.6, 2.4, 1.2 ]]
        >>> #  [[ 1, 1.5, 2 ]
        >>> #   [ 1.5, 1, 1.5 ]
        >>> #   [ 2, 1.5, 1 ]]]
    """

    def __init__(self, lazy_var, constant):
        if torch.is_tensor(constant):
            if constant.ndimension() > 1:
                raise RuntimeError(
                    "Got a constant with %d dimensions - expected a 0D or 1D tensor" % constant.ndimension()
                )
            elif constant.numel() > 1:
                if not (lazy_var.ndimension() == 3 and lazy_var.size(0) == constant.numel()):
                    numel = constant.numel()
                    raise RuntimeError(
                        "A constant with size %d expedts a 3D lazy var. with batch size %d. "
                        "Got a %dD lazy var. with size %s"
                        % (numel, numel, lazy_var.ndimension(), repr(lazy_var.size()))
                    )

            elif constant.numel() == 1:
                constant = constant.squeeze()
        else:
            constant = torch.tensor(constant, device=lazy_var.device, dtype=torch.float32)

        super(ConstantMulLazyVariable, self).__init__(lazy_var, constant)
        self.lazy_var = lazy_var
        self.constant = constant

    def _matmul(self, rhs):
        res = self.lazy_var._matmul(rhs)
        res = res * self._constant_as(res)
        return res

    def _t_matmul(self, rhs):
        res = self.lazy_var._t_matmul(rhs)
        res = res * self._constant_as(res)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        res = list(self.lazy_var._quad_form_derivative(left_vecs, right_vecs))
        for i, item in enumerate(res):
            if torch.is_tensor(item) and res[i].sum():
                res[i] = res[i] * self.constant.expand_as(res[i])
        # Gradient with respect to the constant
        if self.constant.numel() == 1:
            constant_deriv = (left_vecs * self.lazy_var._matmul(right_vecs)).sum().expand_as(self.constant)
        else:
            constant_deriv = (left_vecs * self.lazy_var._matmul(right_vecs)).sum(-2, keepdim=True).sum(-1, keepdim=True)

        res.append(constant_deriv)
        return res

    def _constant_as(self, other):
        size = [self.constant.numel()] + [1] * (other.ndimension() - 1)
        constant = self.constant.view(*size)

        if constant.ndimension() > other.ndimension():
            constant = constant.squeeze(-1)

        return constant

    def _size(self):
        return self.lazy_var.size()

    def _transpose_nonbatch(self):
        return ConstantMulLazyVariable(self.lazy_var._transpose_nonbatch(), self.constant)

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        res = self.lazy_var._batch_get_indices(batch_indices, left_indices, right_indices)
        return self.constant.expand_as(res) * res

    def _get_indices(self, left_indices, right_indices):
        res = self.lazy_var._get_indices(left_indices, right_indices)
        return self.constant.expand_as(res) * res

    def _approx_diag(self):
        res = self.lazy_var._approx_diag()
        return res * self._constant_as(res)

    def evaluate(self):
        res = self.lazy_var.evaluate()
        return res * self._constant_as(res)

    def diag(self):
        res = self.lazy_var.diag()
        res = res * self._constant_as(res)
        return res

    def repeat(self, *sizes):
        return ConstantMulLazyVariable(self.lazy_var.repeat(*sizes), self.constant)

    def __getitem__(self, i):
        constant = self.constant
        if constant.numel() > 1:
            first_index = i[0] if isinstance(i, tuple) else i
            constant = constant[first_index]
        return self.lazy_var.__getitem__(i) * constant
