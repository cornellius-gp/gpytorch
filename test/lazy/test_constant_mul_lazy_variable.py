from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
import gpytorch
from gpytorch.lazy import ToeplitzLazyVariable
from torch.autograd import Variable


class TestConstantMulLazyVariable(unittest.TestCase):

    def test_inv_matmul(self):
        labels_var = Variable(torch.randn(4), requires_grad=True)
        labels_var_copy = Variable(labels_var.data, requires_grad=True)
        grad_output = torch.randn(4)

        # Test case
        c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
        c2_var = Variable(torch.Tensor([12.5, 2.5, 5, 0]), requires_grad=True)
        toeplitz_lazy_var = ToeplitzLazyVariable(c1_var) * 2.5
        actual = ToeplitzLazyVariable(c2_var)

        # Test forward
        with gpytorch.settings.max_cg_iterations(1000):
            res = toeplitz_lazy_var.inv_matmul(labels_var)
            actual = gpytorch.inv_matmul(actual, labels_var_copy)

        # Test backwards
        res.backward(grad_output)
        actual.backward(grad_output)

        self.assertLess(
            math.fabs(res.data.squeeze()[0] - actual.data.squeeze()[0]),
            6e-1,
        )
        self.assertLess(math.fabs(c1_var.grad.data[0] - c2_var.grad.data[0]), 1)

    def test_getitem(self):
        c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
        c2_var = Variable(torch.Tensor([12.5, 2.5, 5, 0]), requires_grad=True)
        toeplitz_lazy_var = ToeplitzLazyVariable(c1_var) * 2.5
        actual = ToeplitzLazyVariable(c2_var)

        diff = (torch.norm(
            actual[2:, 2:].evaluate().data -
            toeplitz_lazy_var[2:, 2:].evaluate().data
        ))
        self.assertLess(diff, 1e-3)


if __name__ == '__main__':
    unittest.main()
