from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
import gpytorch
from torch.autograd import Variable
from torch import nn


class AddDiagTest(unittest.TestCase):

    def test_forward(self):
        a = nn.Parameter(torch.Tensor([5]))
        b = Variable(torch.ones(3, 3))
        output = gpytorch.add_diag(b, a)

        actual = torch.Tensor([
            [6, 1, 1],
            [1, 6, 1],
            [1, 1, 6],
        ])
        self.assertLess(torch.norm(output.data - actual), 1e-7)

    def test_backward(self):
        grad = torch.randn(3, 3)

        a = nn.Parameter(torch.Tensor([3]))
        b = Variable(torch.ones(3, 3), requires_grad=True)
        output = gpytorch.add_diag(b, a)
        output.backward(gradient=grad)

        self.assertLess(math.fabs(a.grad.data[0] - grad.trace()), 1e-6)
        self.assertLess(torch.norm(b.grad.data - grad), 1e-6)


if __name__ == '__main__':
    unittest.main()
