from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gpytorch
import torch
import unittest
from gpytorch import utils
from torch.autograd import Variable
from gpytorch.lazy import ToeplitzLazyVariable


class TestToeplitzLazyVariable(unittest.TestCase):
    def setUp(self):
        self.toeplitz_column = torch.Tensor([2, 0, 4, 1])
        self.batch_toeplitz_column = torch.Tensor([[2, 0, 4, 1], [1, 1, -1, 3]])

    def test_inv_matmul(self):
        c_1 = Variable(torch.Tensor([4, 1, 1]), requires_grad=True)
        c_2 = Variable(torch.Tensor([4, 1, 1]), requires_grad=True)
        T_1 = Variable(torch.zeros(3, 3))
        for i in range(3):
            for j in range(3):
                T_1[i, j] = c_1[abs(i - j)]
        T_2 = gpytorch.lazy.ToeplitzLazyVariable(c_2)

        B = Variable(torch.randn(3, 4))

        res_1 = gpytorch.inv_matmul(T_1, B).sum()
        res_2 = gpytorch.inv_matmul(T_2, B).sum()

        res_1.backward()
        res_2.backward()

        self.assertLess(torch.norm(res_1.data - res_2.data), 1e-4)
        self.assertLess(torch.norm(c_1.grad.data - c_2.grad.data), 1e-4)

    def test_evaluate(self):
        lazy_toeplitz_var = ToeplitzLazyVariable(Variable(self.toeplitz_column))
        res = lazy_toeplitz_var.evaluate()
        actual = torch.Tensor([[2, 0, 4, 1], [0, 2, 0, 4], [4, 0, 2, 0], [1, 4, 0, 2]])
        self.assertTrue(utils.approx_equal(res, actual))

        lazy_toeplitz_var = ToeplitzLazyVariable(Variable(self.batch_toeplitz_column))
        res = lazy_toeplitz_var.evaluate()
        actual = torch.Tensor(
            [
                [[2, 0, 4, 1], [0, 2, 0, 4], [4, 0, 2, 0], [1, 4, 0, 2]],
                [[1, 1, -1, 3], [1, 1, 1, -1], [-1, 1, 1, 1], [3, -1, 1, 1]],
            ]
        )
        self.assertTrue(utils.approx_equal(res, actual))

    def test_get_item_square_on_variable(self):
        toeplitz_var = ToeplitzLazyVariable(Variable(torch.Tensor([1, 2, 3, 4])))
        evaluated = toeplitz_var.evaluate().data

        self.assertTrue(utils.approx_equal(toeplitz_var[2:4, 2:4].evaluate().data, evaluated[2:4, 2:4]))

    def test_get_item_on_batch(self):
        toeplitz_var = ToeplitzLazyVariable(Variable(self.batch_toeplitz_column))
        evaluated = toeplitz_var.evaluate().data
        self.assertTrue(utils.approx_equal(toeplitz_var[0, 1:3].evaluate().data, evaluated[0, 1:3]))

    def test_get_item_scalar_on_batch(self):
        toeplitz_var = ToeplitzLazyVariable(Variable(torch.Tensor([[1, 2, 3, 4]])))
        evaluated = toeplitz_var.evaluate().data
        self.assertTrue(utils.approx_equal(toeplitz_var[0].evaluate().data, evaluated[0]))


if __name__ == "__main__":
    unittest.main()
