import torch
from torch.autograd import Variable
from gpytorch.lazy import MatmulLazyVariable
from gpytorch.utils import approx_equal


def test_diag():
    lhs = Variable(torch.randn(5, 3))
    rhs = Variable(torch.randn(3, 5))
    actual = lhs.matmul(rhs)
    res = MatmulLazyVariable(lhs, rhs)
    assert approx_equal(actual.diag().data, res.diag().data)


def test_evaluate():
    lhs = Variable(torch.randn(5, 3))
    rhs = Variable(torch.randn(3, 5))
    actual = lhs.matmul(rhs)
    res = MatmulLazyVariable(lhs, rhs)
    assert approx_equal(actual.data, res.evaluate().data)
