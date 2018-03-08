import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import MatmulLazyVariable
from gpytorch.utils import approx_equal


class TestMatmulLazyVariable(unittest.TestCase):
    def test_matmul(self):
        lhs = Variable(torch.randn(5, 3), requires_grad=True)
        rhs = Variable(torch.randn(3, 4), requires_grad=True)
        covar = MatmulLazyVariable(lhs, rhs)
        mat = Variable(torch.randn(4, 10))
        res = covar.matmul(mat)

        lhs_clone = Variable(lhs.data.clone(), requires_grad=True)
        rhs_clone = Variable(rhs.data.clone(), requires_grad=True)
        mat_clone = Variable(mat.data.clone())
        actual = lhs_clone.matmul(rhs_clone).matmul(mat_clone)

        self.assertTrue(approx_equal(res.data, actual.data))

        actual.sum().backward()

        res.sum().backward()
        self.assertTrue(approx_equal(lhs.grad.data, lhs_clone.grad.data))
        self.assertTrue(approx_equal(rhs.grad.data, rhs_clone.grad.data))

    def test_diag(self):
        lhs = Variable(torch.randn(5, 3))
        rhs = Variable(torch.randn(3, 5))
        actual = lhs.matmul(rhs)
        res = MatmulLazyVariable(lhs, rhs)
        self.assertTrue(approx_equal(actual.diag().data, res.diag().data))

    def test_batch_diag(self):
        lhs = Variable(torch.randn(4, 5, 3))
        rhs = Variable(torch.randn(4, 3, 5))
        actual = lhs.matmul(rhs)
        actual_diag = torch.cat([
            actual[0].diag().unsqueeze(0),
            actual[1].diag().unsqueeze(0),
            actual[2].diag().unsqueeze(0),
            actual[3].diag().unsqueeze(0),
        ])

        res = MatmulLazyVariable(lhs, rhs)
        self.assertTrue(approx_equal(actual_diag.data, res.diag().data))

    def test_evaluate(self):
        lhs = Variable(torch.randn(5, 3))
        rhs = Variable(torch.randn(3, 5))
        actual = lhs.matmul(rhs)
        res = MatmulLazyVariable(lhs, rhs)
        self.assertTrue(approx_equal(actual.data, res.evaluate().data))

    def test_transpose(self):
        lhs = Variable(torch.randn(5, 3))
        rhs = Variable(torch.randn(3, 5))
        actual = lhs.matmul(rhs)
        res = MatmulLazyVariable(lhs, rhs)
        self.assertTrue(approx_equal(actual.t().data, res.t().evaluate().data))


if __name__ == '__main__':
    unittest.main()
