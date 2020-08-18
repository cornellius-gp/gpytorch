#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import InterpolatedLazyTensor, NonLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase, RectangularLazyTensorTestCase


class TestInterpolatedLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 1
    should_test_sample = True

    def test_quad_form_derivative(self):
        # InterpolatedLazyTensor's representation includes int variables (the interp. indices),
        # so the default derivative doesn't apply
        pass

    def create_lazy_tensor(self):
        left_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]])
        left_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float)
        left_interp_values.requires_grad = True
        right_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]])
        right_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float)
        right_interp_values.requires_grad = True

        base_tensor = torch.randn(6, 6)
        base_tensor = base_tensor.t().matmul(base_tensor)
        base_tensor.requires_grad = True
        base_lazy_tensor = NonLazyTensor(base_tensor)

        return InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

    def evaluate_lazy_tensor(self, lazy_tensor):
        left_matrix = torch.zeros(4, 6)
        right_matrix = torch.zeros(4, 6)
        left_matrix.scatter_(1, lazy_tensor.left_interp_indices, lazy_tensor.left_interp_values)
        right_matrix.scatter_(1, lazy_tensor.right_interp_indices, lazy_tensor.right_interp_values)

        base_tensor = lazy_tensor.base_lazy_tensor.tensor
        actual = left_matrix.matmul(base_tensor).matmul(right_matrix.t())
        return actual


class TestInterpolatedLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def test_quad_form_derivative(self):
        # InterpolatedLazyTensor's representation includes int variables (the interp. indices),
        # so the default derivative doesn't apply
        pass

    def create_lazy_tensor(self):
        left_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1)
        left_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(5, 1, 1)
        left_interp_values.requires_grad = True
        right_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]]).repeat(5, 1, 1)
        right_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(5, 1, 1)
        right_interp_values.requires_grad = True

        base_tensor = torch.randn(5, 6, 6)
        base_tensor = base_tensor.transpose(-2, -1).matmul(base_tensor)
        base_tensor.requires_grad = True
        base_lazy_tensor = NonLazyTensor(base_tensor)

        return InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

    def evaluate_lazy_tensor(self, lazy_tensor):
        left_matrix_comps = []
        right_matrix_comps = []
        for i in range(5):
            left_matrix_comp = torch.zeros(4, 6)
            right_matrix_comp = torch.zeros(4, 6)
            left_matrix_comp.scatter_(1, lazy_tensor.left_interp_indices[i], lazy_tensor.left_interp_values[i])
            right_matrix_comp.scatter_(1, lazy_tensor.right_interp_indices[i], lazy_tensor.right_interp_values[i])
            left_matrix_comps.append(left_matrix_comp.unsqueeze(0))
            right_matrix_comps.append(right_matrix_comp.unsqueeze(0))
        left_matrix = torch.cat(left_matrix_comps)
        right_matrix = torch.cat(right_matrix_comps)

        base_tensor = lazy_tensor.base_lazy_tensor.tensor
        actual = left_matrix.matmul(base_tensor).matmul(right_matrix.transpose(-1, -2))
        return actual


class TestInterpolatedLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def test_quad_form_derivative(self):
        # InterpolatedLazyTensor's representation includes int variables (the interp. indices),
        # so the default derivative doesn't apply
        pass

    def create_lazy_tensor(self):
        left_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]]).repeat(2, 5, 1, 1)
        left_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(2, 5, 1, 1)
        left_interp_values.requires_grad = True
        right_interp_indices = torch.LongTensor([[0, 1], [2, 3], [3, 4], [4, 5]]).repeat(2, 5, 1, 1)
        right_interp_values = torch.tensor([[0.1, 0.9], [1, 2], [0.5, 1], [1, 3]], dtype=torch.float).repeat(2, 5, 1, 1)
        right_interp_values.requires_grad = True

        base_tensor = torch.randn(5, 6, 6)
        base_tensor = base_tensor.transpose(-2, -1).matmul(base_tensor)
        base_lazy_tensor = NonLazyTensor(base_tensor)

        return InterpolatedLazyTensor(
            base_lazy_tensor, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )

    def evaluate_lazy_tensor(self, lazy_tensor):
        left_matrix_comps = []
        right_matrix_comps = []
        for i in range(2):
            for j in range(5):
                left_matrix_comp = torch.zeros(4, 6)
                right_matrix_comp = torch.zeros(4, 6)
                left_matrix_comp.scatter_(
                    1, lazy_tensor.left_interp_indices[i, j], lazy_tensor.left_interp_values[i, j]
                )
                right_matrix_comp.scatter_(
                    1, lazy_tensor.right_interp_indices[i, j], lazy_tensor.right_interp_values[i, j]
                )
                left_matrix_comps.append(left_matrix_comp.unsqueeze(0))
                right_matrix_comps.append(right_matrix_comp.unsqueeze(0))
        left_matrix = torch.cat(left_matrix_comps)
        right_matrix = torch.cat(right_matrix_comps)
        left_matrix = left_matrix.view(2, 5, 4, 6)
        right_matrix = right_matrix.view(2, 5, 4, 6)

        base_tensor = lazy_tensor.base_lazy_tensor.tensor
        actual = left_matrix.matmul(base_tensor).matmul(right_matrix.transpose(-1, -2))
        return actual


def empty_method(self):
    pass


class TestInterpolatedLazyTensorRectangular(RectangularLazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        itplzt = InterpolatedLazyTensor(NonLazyTensor(torch.rand(3, 4)))
        return itplzt

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.base_lazy_tensor.tensor

    # Disable tests meant for square matrices
    test_add_diag = empty_method
    test_diag = empty_method
    test_inv_matmul_matrix = empty_method
    test_inv_matmul_matrix_broadcast = empty_method
    test_inv_matmul_matrix_cholesky = empty_method
    test_inv_matmul_matrix_with_left = empty_method
    test_inv_matmul_vector = empty_method
    test_inv_matmul_vector_with_left = empty_method
    test_inv_matmul_vector_with_left_cholesky = empty_method
    test_inv_quad_logdet = empty_method
    test_inv_quad_logdet_no_reduce = empty_method
    test_inv_quad_logdet_no_reduce_cholesky = empty_method
    test_quad_form_derivative = empty_method
    test_root_decomposition = empty_method
    test_root_decomposition_cholesky = empty_method
    test_root_inv_decomposition = empty_method
    test_sqrt_inv_matmul = empty_method
    test_sqrt_inv_matmul_no_lhs = empty_method
    test_symeig = empty_method
    test_svd = empty_method


if __name__ == "__main__":
    unittest.main()
