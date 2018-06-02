from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import RootLazyVariable
from gpytorch.utils import approx_equal


class TestRootLazyVariable(unittest.TestCase):
    def test_matmul(self):
        root = Variable(torch.randn(5, 3), requires_grad=True)
        covar = RootLazyVariable(root)
        mat = Variable(torch.eye(5))
        res = covar.matmul(mat)

        root_clone = Variable(root.data.clone(), requires_grad=True)
        mat_clone = Variable(mat.data.clone())
        actual = root_clone.matmul(root_clone.transpose(-1, -2)).matmul(mat_clone)

        self.assertTrue(approx_equal(res.data, actual.data))

        gradient = torch.randn(5, 5)
        actual.backward(gradient=Variable(gradient))
        res.backward(gradient=Variable(gradient))

        self.assertTrue(approx_equal(root.grad.data, root_clone.grad.data))

    def test_diag(self):
        root = Variable(torch.randn(5, 3))
        actual = root.matmul(root.transpose(-1, -2))
        res = RootLazyVariable(root)
        self.assertTrue(approx_equal(actual.diag().data, res.diag().data))

    def test_batch_diag(self):
        root = Variable(torch.randn(4, 5, 3))
        actual = root.matmul(root.transpose(-1, -2))
        actual_diag = torch.cat(
            [
                actual[0].diag().unsqueeze(0),
                actual[1].diag().unsqueeze(0),
                actual[2].diag().unsqueeze(0),
                actual[3].diag().unsqueeze(0),
            ]
        )

        res = RootLazyVariable(root)
        self.assertTrue(approx_equal(actual_diag.data, res.diag().data))

    def test_evaluate(self):
        root = Variable(torch.randn(5, 3))
        actual = root.matmul(root.transpose(-1, -2))
        res = RootLazyVariable(root)
        self.assertTrue(approx_equal(actual.data, res.evaluate().data))


if __name__ == "__main__":
    unittest.main()
