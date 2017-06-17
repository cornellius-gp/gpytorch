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
    out_var = InverseQuadForm(Variable(b))(a_var)
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
    actual_grad = torch.ger(a_inv_b, a_inv_b) * 2

    a_var = Variable(a, requires_grad=True)
    out_var = InverseQuadForm(Variable(b))(a_var) 
    out_var = out_var * 2
    out_var.backward()
    res = a_var.grad.data

    print(actual_grad)
    print(a_var.grad.data)

    assert(torch.norm(actual_grad - res) < 1e-4)
