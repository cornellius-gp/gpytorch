from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
import gpytorch
from torch.autograd import Variable


class DSMMTest(unittest.TestCase):
    def test_forward(self):
        i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
        v = torch.FloatTensor([3, 4, 5])
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
        dense = Variable(torch.randn(3, 3))

        res = gpytorch.dsmm(Variable(sparse), dense)
        actual = torch.mm(Variable(sparse.to_dense()), dense)
        self.assertLess(torch.norm(res.data - actual.data), 1e-5)

    def test_forward_batch(self):
        i = torch.LongTensor([[0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [2, 0, 2, 2, 0, 2]])
        v = torch.FloatTensor([3, 4, 5, 6, 7, 8])
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 2, 3]))
        dense = Variable(torch.randn(2, 3, 3))

        res = gpytorch.dsmm(Variable(sparse), dense)
        actual = torch.matmul(Variable(sparse.to_dense()), dense)
        self.assertLess(torch.norm(res.data - actual.data), 1e-5)

    def test_backward(self):
        i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
        v = torch.FloatTensor([3, 4, 5])
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
        dense = Variable(torch.randn(3, 4), requires_grad=True)
        dense_copy = Variable(dense.data.clone(), requires_grad=True)
        grad_output = torch.randn(2, 4)

        res = gpytorch.dsmm(Variable(sparse), dense)
        res.backward(grad_output)
        actual = torch.mm(Variable(sparse.to_dense()), dense_copy)
        actual.backward(grad_output)
        self.assertLess(torch.norm(dense.grad.data - dense_copy.grad.data), 1e-5)

    def test_backward_batch(self):
        i = torch.LongTensor([[0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [2, 0, 2, 2, 0, 2]])
        v = torch.FloatTensor([3, 4, 5, 6, 7, 8])
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 2, 3]))
        dense = Variable(torch.randn(2, 3, 4), requires_grad=True)
        dense_copy = Variable(dense.data.clone(), requires_grad=True)
        grad_output = torch.randn(2, 2, 4)

        res = gpytorch.dsmm(Variable(sparse), dense)
        res.backward(grad_output)
        actual = torch.matmul(Variable(sparse.to_dense()), dense_copy)
        actual.backward(grad_output)
        self.assertLess(torch.norm(dense.grad.data - dense_copy.grad.data), 1e-5)


if __name__ == "__main__":
    unittest.main()
