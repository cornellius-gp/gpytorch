#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import ZeroLazyTensor
from gpytorch.test.utils import approx_equal


class TestZeroLazyTensor(unittest.TestCase):
    def test_evaluate(self):
        lv = ZeroLazyTensor(5, 4, 3)
        actual = torch.zeros(5, 4, 3)
        res = lv.evaluate()
        self.assertLess(torch.norm(res - actual), 1e-4)

    def test_getitem(self):
        lv = ZeroLazyTensor(5, 4, 3)

        res_one = lv[0].evaluate()
        self.assertLess(torch.norm(res_one - torch.zeros(4, 3)), 1e-4)
        res_two = lv[:, 1, :]
        self.assertLess(torch.norm(res_two - torch.zeros(5, 3)), 1e-4)
        res_three = lv[:, :, 2]
        self.assertLess(torch.norm(res_three - torch.zeros(5, 4)), 1e-4)

    def test_getitem_complex(self):
        lv = ZeroLazyTensor(5, 4, 3)

        res_one = lv[[0, 1]].evaluate()
        self.assertLess(torch.norm(res_one - torch.zeros(2, 4, 3)), 1e-4)
        res_two = lv[:, [0, 1], :].evaluate()
        self.assertLess(torch.norm(res_two - torch.zeros(5, 2, 3)), 1e-4)
        res_three = lv[:, :, [0, 2]].evaluate()
        self.assertLess(torch.norm(res_three - torch.zeros(5, 4, 2)), 1e-4)

    def test_getitem_ellipsis(self):
        lv = ZeroLazyTensor(5, 4, 3)

        res_one = lv[[0, 1]].evaluate()
        self.assertLess(torch.norm(res_one - torch.zeros(2, 4, 3)), 1e-4)
        res_two = lv[:, [0, 1], ...].evaluate()
        self.assertLess(torch.norm(res_two - torch.zeros(5, 2, 3)), 1e-4)
        res_three = lv[..., [0, 2]].evaluate()
        self.assertLess(torch.norm(res_three - torch.zeros(5, 4, 2)), 1e-4)

    def test_get_item_tensor_index(self):
        # Tests the default LV.__getitem__ behavior
        lazy_tensor = ZeroLazyTensor(5, 5)
        evaluated = lazy_tensor.evaluate()

        index = (torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index].evaluate(), evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index].evaluate(), evaluated[index]))
        index = (Ellipsis, slice(None, None, None), torch.tensor([0, 0, 1, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index].evaluate(), evaluated[index]))
        index = (Ellipsis, torch.tensor([0, 0, 1, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index].evaluate(), evaluated[index]))

    def test_get_item_tensor_index_on_batch(self):
        # Tests the default LV.__getitem__ behavior
        lazy_tensor = ZeroLazyTensor(3, 5, 5)
        evaluated = lazy_tensor.evaluate()

        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), torch.tensor([1, 2, 0, 1]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1]), slice(None, None, None), torch.tensor([0, 1, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 1]), slice(None, None, None), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index].evaluate(), evaluated[index]))
        index = (slice(None, None, None), torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 2]), slice(None, None, None))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (torch.tensor([0, 0, 1, 0]), slice(None, None, None), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(approx_equal(lazy_tensor[index], evaluated[index]))
        index = (Ellipsis, torch.tensor([0, 1, 1, 0]))
        self.assertTrue(approx_equal(lazy_tensor[index].evaluate(), evaluated[index]))

    def test_add_diag(self):
        diag = torch.tensor(1.5)
        res = ZeroLazyTensor(5, 5).add_diag(diag).evaluate()
        actual = torch.eye(5).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([1.5])
        res = ZeroLazyTensor(5, 5).add_diag(diag).evaluate()
        actual = torch.eye(5).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([1.5, 1.3, 1.2, 1.1, 2.0])
        res = ZeroLazyTensor(5, 5).add_diag(diag).evaluate()
        actual = diag.diag()
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor(1.5)
        res = ZeroLazyTensor(2, 5, 5).add_diag(diag).evaluate()
        actual = torch.eye(5).unsqueeze(0).repeat(2, 1, 1).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([1.5])
        res = ZeroLazyTensor(2, 5, 5).add_diag(diag).evaluate()
        actual = torch.eye(5).unsqueeze(0).repeat(2, 1, 1).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([1.5, 1.3, 1.2, 1.1, 2.0])
        res = ZeroLazyTensor(2, 5, 5).add_diag(diag).evaluate()
        actual = diag.diag().unsqueeze(0).repeat(2, 1, 1)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([[1.5, 1.3, 1.2, 1.1, 2.0], [0, 1, 2, 1, 1]])
        res = ZeroLazyTensor(2, 5, 5).add_diag(diag).evaluate()
        actual = torch.cat([diag[0].diag().unsqueeze(0), diag[1].diag().unsqueeze(0)])
        self.assertTrue(approx_equal(res, actual))


if __name__ == "__main__":
    unittest.main()
