from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
import gpytorch
from gpytorch.lazy import ToeplitzLazyTensor


class TestConstantMulLazyTensor(unittest.TestCase):
    def test_inv_matmul(self):
        labels_var = torch.randn(4, requires_grad=True)
        labels_var_copy = labels_var.clone().detach().requires_grad_(True)
        grad_output = torch.randn(4)

        # Test case
        c1_var = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        c2_var = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        toeplitz_lazy_var = ToeplitzLazyTensor(c1_var) * 2.5
        actual = ToeplitzLazyTensor(c2_var).evaluate() * 2.5

        # Test forward
        with gpytorch.settings.max_cg_iterations(1000):
            res = toeplitz_lazy_var.inv_matmul(labels_var)
            actual = gpytorch.inv_matmul(actual, labels_var_copy)

        # Test backwards
        res.backward(grad_output)
        actual.backward(grad_output)

        for i in range(c1_var.size(0)):
            self.assertLess(math.fabs(res[i].item() - actual[i].item()), 1e-2)
            self.assertLess(math.fabs(c1_var.grad[i].item() - c2_var.grad[i].item()), 1e-2)

    def test_batch_inv_matmul(self):
        labels_var = torch.randn(2, 4, 1, requires_grad=True)
        labels_var_copy = labels_var.clone().detach().requires_grad_(True)
        grad_output = torch.randn(2, 4, 1)

        # Test case
        c1_var = torch.tensor([[5, 1, 2, 0]], dtype=torch.float).repeat(2, 1)
        c2_var = torch.tensor([[5, 1, 2, 0]], dtype=torch.float).repeat(2, 1)
        c1_var.requires_grad = True
        c2_var.requires_grad = True
        toeplitz_lazy_var = ToeplitzLazyTensor(c1_var) * torch.tensor([2.5, 1.])
        actual = ToeplitzLazyTensor(c2_var).evaluate() * torch.tensor([2.5, 1.]).view(2, 1, 1)

        # Test forward
        with gpytorch.settings.max_cg_iterations(1000):
            res = toeplitz_lazy_var.inv_matmul(labels_var)
            actual = gpytorch.inv_matmul(actual, labels_var_copy)

        # Test backwards
        res.backward(grad_output)
        actual.backward(grad_output)

        for i in range(c1_var.size(0)):
            for j in range(c1_var.size(1)):
                self.assertLess(math.fabs(res[i, j].item() - actual[i, j].item()), 1e-2)
                self.assertLess(math.fabs(c1_var.grad[i, j].item() - c2_var.grad[i, j].item()), 1e-2)

    def test_diag(self):
        c1_var = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        c2_var = torch.tensor([12.5, 2.5, 5, 0], dtype=torch.float, requires_grad=True)
        toeplitz_lazy_var = ToeplitzLazyTensor(c1_var) * 2.5
        actual = ToeplitzLazyTensor(c2_var)

        diff = torch.norm(actual.diag() - toeplitz_lazy_var.diag())
        self.assertLess(diff, 1e-3)

    def test_getitem(self):
        c1_var = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        c2_var = torch.tensor([12.5, 2.5, 5, 0], dtype=torch.float, requires_grad=True)
        toeplitz_lazy_var = ToeplitzLazyTensor(c1_var) * 2.5
        actual = ToeplitzLazyTensor(c2_var)

        diff = torch.norm(actual[2:, 2:].evaluate() - toeplitz_lazy_var[2:, 2:].evaluate())
        self.assertLess(diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
