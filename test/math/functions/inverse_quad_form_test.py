import math
import torch
import numpy as np
from torch.autograd import Variable
from gpytorch.math.functions import InverseQuadForm


def test_forward():
    a = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
    ])
    b = torch.randn(3)
    actual = a.inverse().mv(b).dot(b)

    a_var = Variable(a)
    b_var = Variable(b)
    out_var = InverseQuadForm()(a_var, b_var)
    res = out_var.data[0]
    
    assert(math.fabs(actual - res) < 1e-4)


def test_backward():
    a = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
    ])
    b = torch.randn(3)

    a_inv_b = a.inverse().mv(b)
    actual_a_grad = -2 * torch.ger(a_inv_b, a_inv_b)
    actual_b_grad = 4 * a_inv_b

    a_var = Variable(a, requires_grad=True)
    b_var = Variable(b, requires_grad=True)
    out_var = InverseQuadForm()(a_var, b_var)
    out_var = out_var * 2
    out_var.backward()

    a_res = a_var.grad.data
    b_res = b_var.grad.data

    assert(torch.norm(actual_a_grad - a_res) < 1e-4)
    assert(torch.norm(actual_b_grad - b_res) < 1e-4)
