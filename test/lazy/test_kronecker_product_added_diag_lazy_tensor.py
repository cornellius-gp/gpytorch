#!/usr/bin/env python3

import unittest
from unittest import mock

import torch

from gpytorch import settings
from gpytorch.lazy import (
    ConstantDiagLazyTensor,
    DiagLazyTensor,
    KroneckerProductAddedDiagLazyTensor,
    KroneckerProductDiagLazyTensor,
    KroneckerProductLazyTensor,
    NonLazyTensor,
)
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestKroneckerProductAddedDiagLazyTensor(unittest.TestCase, LazyTensorTestCase):
    # this lazy tensor has an explicit inverse so we don't need to run these
    skip_slq_tests = True

    def create_lazy_tensor(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0.5, 1, 0], [0.5, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float)
        d = 0.5 * torch.rand(24, dtype=torch.float)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        d.requires_grad_(True)
        kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        diag_lazy_tensor = DiagLazyTensor(d)
        return KroneckerProductAddedDiagLazyTensor(kp_lazy_tensor, diag_lazy_tensor)

    def evaluate_lazy_tensor(self, lazy_tensor):
        tensor = lazy_tensor._lazy_tensor.evaluate()
        diag = lazy_tensor._diag_tensor._diag
        return tensor + diag.diag()

class TestKroneckerProductAddedKroneckerDiagLazyTensor(TestKroneckerProductAddedDiagLazyTensor):
    # this lazy tensor has an explicit inverse so we don't need to run these
    skip_slq_tests = True
    should_call_cg = False
    should_call_lanczos = False

    def create_lazy_tensor(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0.5, 1, 0], [0.5, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float)
        d = torch.tensor([2, 1, 3], dtype=torch.float)
        e = torch.tensor([5], dtype=torch.float) 
        f = torch.tensor([2.5], dtype=torch.float)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        d.requires_grad_(True)
        e.requires_grad_(True)
        f.requires_grad_(True)
        kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        diag_lazy_tensor = KroneckerProductDiagLazyTensor(
            DiagLazyTensor(d), ConstantDiagLazyTensor(e, diag_shape=2), ConstantDiagLazyTensor(f, diag_shape=4)
        )       
        return KroneckerProductAddedDiagLazyTensor(kp_lazy_tensor, diag_lazy_tensor)


class TestKroneckerProductAddedConstDiagLazyTensor(TestKroneckerProductAddedDiagLazyTensor):
    should_call_lanczos = False
    should_call_cg = False

    def create_lazy_tensor(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0.5, 1, 0], [0.5, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        diag_lazy_tensor = ConstantDiagLazyTensor(
            torch.tensor([0.25], dtype=torch.float, requires_grad=True), kp_lazy_tensor.shape[-1],
        )
        return KroneckerProductAddedDiagLazyTensor(kp_lazy_tensor, diag_lazy_tensor)

    def test_if_cholesky_used(self):
        lazy_tensor = self.create_lazy_tensor()
        rhs = torch.randn(lazy_tensor.size(-1))
        # Check that cholesky is not called
        with mock.patch.object(lazy_tensor, "cholesky") as chol_mock:
            self._test_inv_matmul(rhs, cholesky=False)
            chol_mock.assert_not_called()

    def test_root_inv_decomposition_no_cholesky(self):
        with settings.max_cholesky_size(0):
            lazy_tensor = self.create_lazy_tensor()
            test_mat = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 5)
            # Check that cholesky is not called
            with mock.patch.object(lazy_tensor, "cholesky") as chol_mock:
                root_approx = lazy_tensor.root_inv_decomposition()
                res = root_approx.matmul(test_mat)
                actual = torch.solve(test_mat, lazy_tensor.evaluate()).solution
                self.assertAllClose(res, actual, rtol=0.05, atol=0.02)
                chol_mock.assert_not_called()

# class TestKroneckerProductAddedDiagAddedDiagLazyTensor(unittest.TestCase, LazyTensorTestCase):
#     # this lazy tensor has an explicit inverse so we don't need to run these
#     skip_slq_tests = True

#     def create_lazy_tensor(self):
#         a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
#         b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
#         d = torch.tensor([2, 1, 3], dtype=torch.float)
#         e = torch.tensor([5], dtype=torch.float)
#         f = torch.tensor([0.25], dtype=torch.float)
#         a.requires_grad_(True)
#         b.requires_grad_(True)
#         d.requires_grad_(True)
#         e.requires_grad_(True)
#         f.requires_grad_(True)

#         kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b))
#         diag_lazy_tensor = KroneckerProductDiagLazyTensor(
#             DiagLazyTensor(d), ConstantDiagLazyTensor(e, diag_shape=2)
#         )
#         second_diag_lazy_tensor = ConstantDiagLazyTensor(f, diag_shape=6)
#         kpadlt1 = KroneckerProductAddedDiagLazyTensor(kp_lazy_tensor, second_diag_lazy_tensor)
#         return KroneckerProductAddedDiagLazyTensor(kpadlt1, diag_lazy_tensor)

#     def evaluate_lazy_tensor(self, lazy_tensor):
#         lt1_eval = lazy_tensor.lazy_tensor.evaluate()
#         lt2_eval = lazy_tensor.diag_tensor._diag.diag()
#         return lt1_eval + lt2_eval

if __name__ == "__main__":
    unittest.main()
