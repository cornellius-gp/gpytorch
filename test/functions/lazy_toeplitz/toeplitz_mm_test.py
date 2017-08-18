import torch
from gpytorch.functions.lazy_toeplitz import ToeplitzMM
from gpytorch import utils
from torch.autograd import Variable


def test_toeplitz_mm_forward():
    c = Variable(torch.randn(5))
    r = Variable(torch.randn(5))
    r.data[0:1].fill_(c.data[0])
    M = Variable(torch.randn(5, 3))

    T = Variable(utils.toeplitz.toeplitz(c.data, r.data))
    actual = torch.mm(T, M)

    res = ToeplitzMM()(c, r, M)
    assert utils.approx_equal(actual, res)


def test_toeplitz_mm_forward_symmetric():
    c = Variable(torch.randn(5))
    M = Variable(torch.randn(5, 3))

    T = Variable(utils.toeplitz.sym_toeplitz(c.data))
    actual = torch.mm(T, M)

    res = ToeplitzMM()(c, c, M)
    assert utils.approx_equal(actual, res)


def test_toeplitz_mm_backward():
    c = Variable(torch.randn(5), requires_grad=True)
    r = Variable(torch.randn(5), requires_grad=True)
    r.data[0:1].fill_(c.data[0])
    M = Variable(torch.randn(5, 3), requires_grad=True)

    T = Variable(utils.toeplitz.toeplitz(c.data, r.data), requires_grad=True)
    actual = torch.mm(T, M).sum()
    actual.backward()

    actual_M_grad = M.grad.data.clone()
    actual_cr_grad = utils.rcumsum(T.grad.data[0])

    M.grad.data.fill_(0)

    res = ToeplitzMM()(c, r, M).sum()
    res.backward()

    assert utils.approx_equal(M.grad.data, actual_M_grad)
    assert utils.approx_equal(r.grad.data, actual_cr_grad)
    assert utils.approx_equal(c.grad.data, actual_cr_grad)
