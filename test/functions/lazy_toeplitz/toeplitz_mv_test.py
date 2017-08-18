import torch
from gpytorch.functions.lazy_toeplitz import ToeplitzMV
from gpytorch import utils
from torch.autograd import Variable


def test_mv_performs_toeplitz_matrix_vector_multiplication():
    c = Variable(torch.randn(5))
    r = Variable(torch.randn(5))
    r.data[0:1].fill_(c.data[0])
    v = Variable(torch.randn(5))

    m = Variable(utils.toeplitz.toeplitz(c.data, r.data))
    actual = torch.mv(m, v)

    res = ToeplitzMV()(c, r, v)
    assert utils.approx_equal(actual, res)


def test_sym_mv_performs_toeplitz_matrix_vector_multiplication():
    c = Variable(torch.randn(5))
    v = Variable(torch.randn(5))

    m = Variable(utils.toeplitz.sym_toeplitz(c.data))
    actual = torch.mv(m, v)

    res = ToeplitzMV()(c, c, v)
    assert utils.approx_equal(actual, res)


def test_mv_backwards_performs_toeplitz_matrix_vector_multiplication():
    c = Variable(torch.randn(5), requires_grad=True)
    r = Variable(torch.randn(5), requires_grad=True)
    r.data[0:1].fill_(c.data[0])
    v = Variable(torch.randn(5), requires_grad=True)

    m = Variable(utils.toeplitz.toeplitz(c.data, r.data), requires_grad=True)
    actual = torch.mv(m, v).sum(0)
    actual.backward()

    actual_v_grad = v.grad.data.clone()
    actual_cr_grad = utils.rcumsum(m.grad.data[0])

    v.grad.data.fill_(0)

    res = ToeplitzMV()(c, r, v).sum(0)
    res.backward()

    assert utils.approx_equal(v.grad.data, actual_v_grad)
    assert utils.approx_equal(r.grad.data, actual_cr_grad)
    assert utils.approx_equal(c.grad.data, actual_cr_grad)
