#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import LazyTensor, RootLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


def make_random_mat(size, rank, batch_shape=torch.Size(())):
    res = torch.randn(*batch_shape, size, rank)
    return res


class TestMulLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 10

    def create_lazy_tensor(self):
        mat1 = make_random_mat(6, 6)
        mat2 = make_random_mat(6, 6)
        res = RootLazyTensor(mat1) * RootLazyTensor(mat2)
        return res.add_diag(torch.tensor(2.0))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag_tensor = lazy_tensor._diag_tensor.evaluate()
        res = torch.mul(
            lazy_tensor._lazy_tensor.left_lazy_tensor.evaluate(), lazy_tensor._lazy_tensor.right_lazy_tensor.evaluate()
        )
        res = res + diag_tensor
        return res

    def test_quad_form_derivative(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor._diag_tensor.requires_grad_(False)
        lazy_tensor_clone = lazy_tensor.clone().detach_().requires_grad_(True)
        lazy_tensor_clone._diag_tensor.requires_grad_(False)
        left_vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-2), 2)
        right_vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 2)
        deriv_custom = lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        deriv_auto = LazyTensor._quad_form_derivative(lazy_tensor_clone, left_vecs, right_vecs)

        for dc, da in zip(deriv_custom, deriv_auto):
            if dc is not None or da is not None:
                self.assertAllClose(dc, da)


class TestMulLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 2

    def create_lazy_tensor(self):
        mat1 = make_random_mat(6, rank=6, batch_shape=torch.Size((2,)))
        mat2 = make_random_mat(6, rank=6, batch_shape=torch.Size((2,)))
        res = RootLazyTensor(mat1) * RootLazyTensor(mat2)
        return res.add_diag(torch.tensor(2.0))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag_tensor = lazy_tensor._diag_tensor.evaluate()
        res = torch.mul(
            lazy_tensor._lazy_tensor.left_lazy_tensor.evaluate(), lazy_tensor._lazy_tensor.right_lazy_tensor.evaluate()
        )
        res = res + diag_tensor
        return res

    def test_quad_form_derivative(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor._diag_tensor.requires_grad_(False)
        lazy_tensor_clone = lazy_tensor.clone().detach_().requires_grad_(True)
        lazy_tensor_clone._diag_tensor.requires_grad_(False)
        left_vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-2), 2)
        right_vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 2)
        deriv_custom = lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        deriv_auto = LazyTensor._quad_form_derivative(lazy_tensor_clone, left_vecs, right_vecs)

        for dc, da in zip(deriv_custom, deriv_auto):
            if dc is not None or da is not None:
                self.assertAllClose(dc, da)


class TestMulLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 1
    skip_slq_tests = True

    def create_lazy_tensor(self):
        mat1 = make_random_mat(6, rank=6, batch_shape=torch.Size((2, 3)))
        mat2 = make_random_mat(6, rank=6, batch_shape=torch.Size((2, 3)))
        res = RootLazyTensor(mat1) * RootLazyTensor(mat2)
        return res.add_diag(torch.tensor(0.5))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag_tensor = lazy_tensor._diag_tensor.evaluate()
        res = torch.mul(
            lazy_tensor._lazy_tensor.left_lazy_tensor.evaluate(), lazy_tensor._lazy_tensor.right_lazy_tensor.evaluate()
        )
        res = res + diag_tensor
        return res

    def test_inv_quad_logdet(self):
        pass

    def test_quad_form_derivative(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor._diag_tensor.requires_grad_(False)
        lazy_tensor_clone = lazy_tensor.clone().detach_().requires_grad_(True)
        lazy_tensor_clone._diag_tensor.requires_grad_(False)
        left_vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-2), 2)
        right_vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 2)
        deriv_custom = lazy_tensor._quad_form_derivative(left_vecs, right_vecs)
        deriv_auto = LazyTensor._quad_form_derivative(lazy_tensor_clone, left_vecs, right_vecs)

        for dc, da in zip(deriv_custom, deriv_auto):
            if dc is not None or da is not None:
                self.assertAllClose(dc, da)


if __name__ == "__main__":
    unittest.main()
