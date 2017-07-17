import torch
import gpytorch
from torch.autograd import Variable


def test_forward():
    a = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
    ])
    b = torch.randn(3)
    actual = a.inverse().mv(b)

    a_var = Variable(a)
    b_var = Variable(b)
    out_var = gpytorch.invmv(a_var, b_var)
    res = out_var.data

    assert(torch.norm(actual - res) < 1e-4)


def test_backward():
    a = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
    ])
    b = torch.ones(3, 3).fill_(2)
    c = torch.randn(3)
    actual_a_grad = -torch.ger(a.inverse().mul_(0.5).mv(torch.ones(3)), a.inverse().mul_(0.5).mv(c)) * 2 * 2
    actual_c_grad = (a.inverse() / 2).t().mv(torch.ones(3)) * 2

    a_var = Variable(a, requires_grad=True)
    c_var = Variable(c, requires_grad=True)
    out_var = a_var.mul(Variable(b))
    out_var = gpytorch.invmv(out_var, c_var)
    out_var = out_var.sum() * 2
    out_var.backward()
    a_res = a_var.grad.data
    c_res = c_var.grad.data

    assert(torch.norm(actual_a_grad - a_res) < 1e-4)
    assert(torch.norm(actual_c_grad - c_res) < 1e-4)
