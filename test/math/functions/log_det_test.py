import math
import torch
import numpy as np
from torch.autograd import Variable
from gpytorch.math.functions import LogDet


def test_log_det_forward():
    a = torch.randn(3, 3)
    a = a.transpose(0, 1).mm(a) + torch.eye(3) # Make it PD
    a_var = Variable(a)

    actual = math.log(np.linalg.det(a.numpy() + 2))
    out_var = a_var + 2
    out_var = LogDet()(out_var)
    res = out_var.data[0]
    
    assert(math.fabs(actual - res) < 1e-4)


def test_log_det_backward():
    a = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
    ])
    b = torch.ones(3, 3).fill_(2)
    actual_grad = (a).inverse()

    a_var = Variable(a, requires_grad=True)
    out_var = a_var.mul(Variable(b))
    out_var = LogDet()(out_var)
    out_var = out_var
    out_var.backward()
    res = a_var.grad.data

    print(actual_grad)
    print(a_var.grad.data)

    assert(torch.norm(actual_grad - res) < 1e-4)
