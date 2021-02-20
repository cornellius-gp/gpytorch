#!/usr/bin/env python3

import torch

from ..utils.broadcasting import _matmul_broadcast_shape
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .root_lazy_tensor import RootLazyTensor


class MulLazyTensor(LazyTensor):
    def _check_args(self, left_lazy_tensor, right_lazy_tensor):
        if not isinstance(left_lazy_tensor, LazyTensor) or not isinstance(right_lazy_tensor, LazyTensor):
            return "MulLazyTensor expects two LazyTensors."
        if left_lazy_tensor.shape != right_lazy_tensor.shape:
            return "MulLazyTensor expects two LazyTensors of the same size: got {} and {}.".format(
                left_lazy_tensor, right_lazy_tensor
            )

    def __init__(self, left_lazy_tensor, right_lazy_tensor):
        """
        Args:
            - lazy_tensors (A list of LazyTensor) - A list of LazyTensor to multiplicate with.
        """
        if not isinstance(left_lazy_tensor, RootLazyTensor):
            left_lazy_tensor = left_lazy_tensor.root_decomposition()
        if not isinstance(right_lazy_tensor, RootLazyTensor):
            right_lazy_tensor = right_lazy_tensor.root_decomposition()
        super(MulLazyTensor, self).__init__(left_lazy_tensor, right_lazy_tensor)
        self.left_lazy_tensor = left_lazy_tensor
        self.right_lazy_tensor = right_lazy_tensor

    def _get_indices(self, row_index, col_index, *batch_indices):
        left_res = self.left_lazy_tensor._get_indices(row_index, col_index, *batch_indices)
        right_res = self.right_lazy_tensor._get_indices(row_index, col_index, *batch_indices)
        return left_res * right_res

    def _matmul(self, rhs):
        output_shape = _matmul_broadcast_shape(self.shape, rhs.shape)
        output_batch_shape = output_shape[:-2]

        is_vector = False
        if rhs.ndimension() == 1:
            rhs = rhs.unsqueeze(1)
            is_vector = True

        # Here we have a root decomposition
        if isinstance(self.left_lazy_tensor, RootLazyTensor):
            left_root = self.left_lazy_tensor.root.evaluate()
            left_res = rhs.unsqueeze(-2) * left_root.unsqueeze(-1)

            rank = left_root.size(-1)
            n = self.size(-1)
            m = rhs.size(-1)
            # Now implement the formula (A . B) v = diag(A D_v B)
            left_res = left_res.view(*output_batch_shape, n, rank * m)
            left_res = self.right_lazy_tensor._matmul(left_res)
            left_res = left_res.view(*output_batch_shape, n, rank, m)
            res = left_res.mul_(left_root.unsqueeze(-1)).sum(-2)
        # This is the case where we're not doing a root decomposition, because the matrix is too small
        else:
            res = (self.left_lazy_tensor.evaluate() * self.right_lazy_tensor.evaluate()).matmul(rhs)
        res = res.squeeze(-1) if is_vector else res
        return res

    def _mul_constant(self, constant):
        if constant > 0:
            res = self.__class__(self.left_lazy_tensor._mul_constant(constant), self.right_lazy_tensor)
        else:
            # Negative constants can screw up the root_decomposition
            # So we'll do a standard _mul_constant
            res = super()._mul_constant(constant)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)

        *batch_shape, n, num_vecs = left_vecs.size()

        if isinstance(self.right_lazy_tensor, RootLazyTensor):
            right_root = self.right_lazy_tensor.root.evaluate()
            left_factor = left_vecs.unsqueeze(-2) * right_root.unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * right_root.unsqueeze(-1)
            right_rank = right_root.size(-1)
        else:
            right_rank = n
            eye = torch.eye(n, dtype=self.right_lazy_tensor.dtype, device=self.right_lazy_tensor.device)
            left_factor = left_vecs.unsqueeze(-2) * self.right_lazy_tensor.evaluate().unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * eye.unsqueeze(-1)

        left_factor = left_factor.view(*batch_shape, n, num_vecs * right_rank)
        right_factor = right_factor.view(*batch_shape, n, num_vecs * right_rank)
        left_deriv_args = self.left_lazy_tensor._quad_form_derivative(left_factor, right_factor)

        if isinstance(self.left_lazy_tensor, RootLazyTensor):
            left_root = self.left_lazy_tensor.root.evaluate()
            left_factor = left_vecs.unsqueeze(-2) * left_root.unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * left_root.unsqueeze(-1)
            left_rank = left_root.size(-1)
        else:
            left_rank = n
            eye = torch.eye(n, dtype=self.left_lazy_tensor.dtype, device=self.left_lazy_tensor.device)
            left_factor = left_vecs.unsqueeze(-2) * self.left_lazy_tensor.evaluate().unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * eye.unsqueeze(-1)

        left_factor = left_factor.view(*batch_shape, n, num_vecs * left_rank)
        right_factor = right_factor.view(*batch_shape, n, num_vecs * left_rank)
        right_deriv_args = self.right_lazy_tensor._quad_form_derivative(left_factor, right_factor)

        return tuple(list(left_deriv_args) + list(right_deriv_args))

    def _expand_batch(self, batch_shape):
        return self.__class__(
            self.left_lazy_tensor._expand_batch(batch_shape), self.right_lazy_tensor._expand_batch(batch_shape)
        )

    def diag(self):
        res = self.left_lazy_tensor.diag() * self.right_lazy_tensor.diag()
        return res

    @cached
    def evaluate(self):
        return self.left_lazy_tensor.evaluate() * self.right_lazy_tensor.evaluate()

    def _size(self):
        return self.left_lazy_tensor.size()

    def _transpose_nonbatch(self):
        # mul.lazy_tensor only works with symmetric matrices
        return self

    def representation(self):
        """
        Returns the Tensors that are used to define the LazyTensor
        """
        res = super(MulLazyTensor, self).representation()
        return res

    def representation_tree(self):
        return super(MulLazyTensor, self).representation_tree()
