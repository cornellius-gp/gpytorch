from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import DiagLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase, BatchLazyTensorTestCase


class TestDiagLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        diag = torch.tensor([1., 2., 4., 2., 3.], requires_grad=True)
        return DiagLazyTensor(diag)

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag
        return diag.diag()


class TestDiagLazyTensorBatch(BatchLazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        diag = torch.tensor([[1., 2., 4., 2., 3.], [2., 1., 2., 1., 4.], [1., 2., 2., 3., 4.]], requires_grad=True)
        return DiagLazyTensor(diag)

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag
        return torch.cat([diag[i].diag().unsqueeze(0) for i in range(3)])

    def test_add_diag(self):
        other_diag = torch.tensor(1.5)
        res = DiagLazyTensor(diag).add_diag(other_diag).evaluate()
        actual = diag.diag() + torch.eye(3).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.tensor([1.5])
        res = DiagLazyTensor(diag).add_diag(other_diag).evaluate()
        actual = diag.diag() + torch.eye(3).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.tensor([1.5, 1.3, 1.2])
        res = DiagLazyTensor(diag).add_diag(other_diag).evaluate()
        actual = diag.diag() + other_diag.diag()
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.tensor(1.5)
        res = DiagLazyTensor(diag.unsqueeze(0).repeat(2, 1)).add_diag(other_diag).evaluate()
        actual = diag.diag().unsqueeze(0) + torch.eye(3).unsqueeze(0).repeat(2, 1, 1).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.tensor([1.5])
        res = DiagLazyTensor(diag.unsqueeze(0).repeat(2, 1)).add_diag(other_diag).evaluate()
        actual = diag.diag().unsqueeze(0) + torch.eye(3).unsqueeze(0).repeat(2, 1, 1).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.tensor([1.5, 1.3, 1.2])
        res = DiagLazyTensor(diag.unsqueeze(0).repeat(2, 1)).add_diag(other_diag).evaluate()
        actual = diag.diag().unsqueeze(0) + other_diag.diag().unsqueeze(0).repeat(2, 1, 1)
        self.assertTrue(approx_equal(res, actual))

        other_diag = torch.tensor([[1.5, 1.3, 1.2], [0, 1, 2]])
        res = DiagLazyTensor(diag.unsqueeze(0).repeat(2, 1)).add_diag(other_diag).evaluate()
        actual = diag.diag().unsqueeze(0)
        actual = actual + torch.cat([other_diag[0].diag().unsqueeze(0), other_diag[1].diag().unsqueeze(0)])
        self.assertTrue(approx_equal(res, actual))


if __name__ == "__main__":
    unittest.main()
