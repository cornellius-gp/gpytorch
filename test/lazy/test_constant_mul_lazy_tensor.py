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
        labels_var = torch.tensor(torch.randn(4), requires_grad=True)
        labels_var_copy = torch.tensor(labels_var, requires_grad=True)
        grad_output = torch.randn(4)

        # Test case
        c1_var = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        c2_var = torch.tensor([12.5, 2.5, 5, 0], dtype=torch.float, requires_grad=True)
        toeplitz_lazy_var = ToeplitzLazyTensor(c1_var) * 2.5
        actual = ToeplitzLazyTensor(c2_var)

        # Test forward
        with gpytorch.settings.max_cg_iterations(1000):
            res = toeplitz_lazy_var.inv_matmul(labels_var)
            actual = gpytorch.inv_matmul(actual, labels_var_copy)

        # Test backwards
        res.backward(grad_output)
        actual.backward(grad_output)

        for i in range(len(c1_var.size())):
            self.assertLess(math.fabs(res[i].item() - actual[i].item()), 6e-1)
            self.assertLess(math.fabs(c1_var.grad[i].item() - c2_var.grad[i].item()), 1)

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

        diff = torch.norm(actual[2:, 2:].evaluate().data - toeplitz_lazy_var[2:, 2:].evaluate().data)
        self.assertLess(diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
