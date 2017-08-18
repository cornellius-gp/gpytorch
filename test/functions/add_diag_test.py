import math
import torch
import gpytorch
from torch.autograd import Variable
from torch import nn


def test_forward():
    a = nn.Parameter(torch.Tensor([5]))
    b = Variable(torch.ones(3, 3))
    output = gpytorch.add_diag(b, a)

    actual = torch.Tensor([
        [6, 1, 1],
        [1, 6, 1],
        [1, 1, 6],
    ])
    assert(torch.norm(output.data - actual) < 1e-7)


def test_backward():
    grad = torch.randn(3, 3)

    a = nn.Parameter(torch.Tensor([3]))
    b = Variable(torch.ones(3, 3), requires_grad=True)
    output = gpytorch.add_diag(b, a)
    output.backward(gradient=grad)

    assert(math.fabs(a.grad.data[0] - grad.trace()) < 1e-6)
    assert(torch.norm(b.grad.data - grad) < 1e-6)
