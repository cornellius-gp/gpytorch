import math
import torch
from torch.nn import Parameter
from gpytorch.math.functions import Diag

def test_forward():
    a = Parameter(torch.Tensor([3])) 
    output = Diag(3)(a)

    actual = torch.Tensor([
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
    ])
    assert(torch.norm(output.data - actual) < 1e-7)


def test_backward():
    grad = torch.randn(3, 3)

    a = Parameter(torch.Tensor([3])) 
    output = Diag(3)(a)
    output.backward(gradient=grad)

    assert(math.fabs(a.grad.data[0] - grad.trace()) < 1e-6)
