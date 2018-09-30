from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gpytorch
import torch
import unittest

import test._utils
from gpytorch.lazy import ToeplitzLazyTensor


class TestToeplitzLazyTensor(unittest.TestCase):
    def setUp(self):
        self.toeplitz_column = torch.tensor([2, 0, 4, 1], dtype=torch.float)
        self.batch_toeplitz_column = torch.tensor([[2, 0, 4, 1], [1, 1, -1, 3]], dtype=torch.float)

    def test_inv_matmul(self):
        c_1 = torch.tensor([4, 1, 1], dtype=torch.float, requires_grad=True)
        c_2 = torch.tensor([4, 1, 1], dtype=torch.float, requires_grad=True)
        T_1 = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                T_1[i, j] = c_1[abs(i - j)]
        T_2 = gpytorch.lazy.ToeplitzLazyTensor(c_2)

        B = torch.randn(3, 4)

        res_1 = gpytorch.inv_matmul(T_1, B).sum()
        res_2 = gpytorch.inv_matmul(T_2, B).sum()

        res_1.backward()
        res_2.backward()

        self.assertLess(torch.norm(res_1 - res_2), 1e-4)
        self.assertLess(torch.norm(c_1.grad - c_2.grad), 1e-4)

    def test_evaluate(self):
        lazy_toeplitz_var = ToeplitzLazyTensor(self.toeplitz_column)
        res = lazy_toeplitz_var.evaluate()
        actual = torch.tensor([[2, 0, 4, 1], [0, 2, 0, 4], [4, 0, 2, 0], [1, 4, 0, 2]], dtype=torch.float)
        self.assertTrue(test._utils.approx_equal(res, actual))

        lazy_toeplitz_var = ToeplitzLazyTensor(self.batch_toeplitz_column)
        res = lazy_toeplitz_var.evaluate()
        actual = torch.tensor(
            [
                [[2, 0, 4, 1], [0, 2, 0, 4], [4, 0, 2, 0], [1, 4, 0, 2]],
                [[1, 1, -1, 3], [1, 1, 1, -1], [-1, 1, 1, 1], [3, -1, 1, 1]],
            ],
            dtype=torch.float,
        )
        self.assertTrue(test._utils.approx_equal(res, actual))

    def test_get_item_square_on_tensor(self):
        # Tests the default LV.__getitem__ behavior
        toeplitz_var = ToeplitzLazyTensor(torch.tensor([1, 2, 3, 4], dtype=torch.float))
        evaluated = toeplitz_var.evaluate()
        self.assertTrue(test._utils.approx_equal(toeplitz_var[2:4, 2:4].evaluate(), evaluated[2:4, 2:4]))

    def test_get_item_tensor_index(self):
        # Tests the default LV.__getitem__ behavior
        toeplitz_var = ToeplitzLazyTensor(torch.tensor([1, 2, 3, 4], dtype=torch.float))
        evaluated = toeplitz_var.evaluate()
        index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index], evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index], evaluated[index]))

    def test_get_item_on_batch(self):
        # Tests the default LV.__getitem__ behavior
        toeplitz_var = ToeplitzLazyTensor(self.batch_toeplitz_column)
        evaluated = toeplitz_var.evaluate()
        self.assertTrue(test._utils.approx_equal(toeplitz_var[0, 1:3].evaluate(), evaluated[0, 1:3]))

    def test_get_item_scalar_on_batch(self):
        # Tests the default LV.__getitem__ behavior
        toeplitz_var = ToeplitzLazyTensor(torch.tensor([[1, 2, 3, 4]], dtype=torch.float))
        evaluated = toeplitz_var.evaluate()
        self.assertTrue(test._utils.approx_equal(toeplitz_var[0].evaluate(), evaluated[0]))

    def test_get_item_tensor_index_on_batch(self):
        # Tests the default LV.__getitem__ behavior
        toeplitz_var = ToeplitzLazyTensor(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float))
        evaluated = toeplitz_var.evaluate()
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 3]))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1]), slice(None, None, None), torch.tensor([0, 1, 2]))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index], evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 1]), slice(None, None, None), slice(None, None, None))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index].evaluate(), evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 0]), slice(None, None, None), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(test._utils.approx_equal(toeplitz_var[index], evaluated[index]))


if __name__ == "__main__":
    unittest.main()
