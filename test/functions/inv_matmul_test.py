import torch
import gpytorch
from torch.autograd import Variable


def test_forward_mm():
    for n_cols in [2, 3, 4]:
        a = torch.Tensor([
            [5, -3, 0],
            [-3, 5, 0],
            [0, 0, 2],
        ])
        b = torch.randn(3, n_cols)
        actual = a.inverse().mm(b)

        a_var = Variable(a)
        b_var = Variable(b)
        out_var = gpytorch.inv_matmul(a_var, b_var)
        res = out_var.data

        assert(torch.norm(actual - res) < 1e-4)


def test_backward_mm():
    for n_cols in [2, 3, 4]:
        a = torch.Tensor([
            [5, -3, 0],
            [-3, 5, 0],
            [0, 0, 2],
        ])
        b = torch.ones(3, 3).fill_(2)
        c = torch.randn(3, n_cols)
        actual_a_grad = -torch.mm(
            a.inverse().mul_(0.5).mm(torch.eye(3, n_cols)),
            a.inverse().mul_(0.5).mm(c).t()
        ) * 2 * 2
        actual_c_grad = (a.inverse() / 2).t().mm(torch.eye(3, n_cols)) * 2

        a_var = Variable(a, requires_grad=True)
        c_var = Variable(c, requires_grad=True)
        out_var = a_var.mul(Variable(b))
        out_var = gpytorch.inv_matmul(out_var, c_var)
        out_var = out_var.mul(Variable(torch.eye(3, n_cols))).sum() * 2
        out_var.backward()
        a_res = a_var.grad.data
        c_res = c_var.grad.data

        assert(torch.norm(actual_a_grad - a_res) < 1e-4)
        assert(torch.norm(actual_c_grad - c_res) < 1e-4)


def test_forward_mv():
    a = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
    ])
    b = torch.randn(3)
    actual = a.inverse().mv(b)

    a_var = Variable(a)
    b_var = Variable(b)
    out_var = gpytorch.inv_matmul(a_var, b_var)
    res = out_var.data

    assert(torch.norm(actual - res) < 1e-4)


def test_backward_mv():
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
    out_var = gpytorch.inv_matmul(out_var, c_var)
    out_var = out_var.sum() * 2
    out_var.backward()
    a_res = a_var.grad.data
    c_res = c_var.grad.data

    assert(torch.norm(actual_a_grad - a_res) < 1e-4)
    assert(torch.norm(actual_c_grad - c_res) < 1e-4)


def test_inv_matmul_azy():
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

    assert(torch.norm(res_1.data - res_2.data) < 1e-4)
    assert(torch.norm(c_1.grad.data - c_2.grad.data) < 1e-4)
