import torch
import gpytorch
from torch.autograd import Variable


def test_forward():
    i = torch.LongTensor([[0, 1, 1],
                          [2, 0, 2]])
    v = torch.FloatTensor([3, 4, 5])
    sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
    dense = Variable(torch.randn(3, 3))

    res = gpytorch.dsmm(Variable(sparse), dense)
    actual = torch.mm(Variable(sparse.to_dense()), dense)
    assert(torch.norm(res.data - actual.data) < 1e-5)


def test_backward():
    i = torch.LongTensor([[0, 1, 1],
                          [2, 0, 2]])
    v = torch.FloatTensor([3, 4, 5])
    sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
    dense = Variable(torch.randn(3, 4), requires_grad=True)
    dense_copy = Variable(dense.data.clone(), requires_grad=True)
    grad_output = torch.randn(2, 4)

    res = gpytorch.dsmm(Variable(sparse), dense)
    res.backward(grad_output)
    actual = torch.mm(Variable(sparse.to_dense()), dense_copy)
    actual.backward(grad_output)
    assert(torch.norm(dense.grad.data - dense_copy.grad.data) < 1e-5)
