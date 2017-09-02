import torch
import gpytorch
from torch.autograd import Variable


def test_matmul_lazy():
    c_1 = Variable(torch.Tensor([4, 1, 1]), requires_grad=True)
    c_2 = Variable(torch.Tensor([4, 1, 1]), requires_grad=True)
    T_1 = Variable(torch.zeros(3, 3))
    for i in range(3):
        for j in range(3):
            T_1[i, j] = c_1[abs(i - j)]
    T_2 = gpytorch.lazy.ToeplitzLazyVariable(c_2)

    B = Variable(torch.randn(3, 4))

    res_1 = T_1.matmul(B).sum()
    res_2 = T_2.matmul(B).sum()

    res_1.backward()
    res_2.backward()

    assert(torch.norm(res_1.data - res_2.data) < 1e-4)
    assert(torch.norm(c_1.grad.data - c_2.grad.data) < 1e-4)
