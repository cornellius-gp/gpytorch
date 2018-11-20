#!/usr/bin/env python3

import math
import torch
import unittest
from gpytorch.kernels import MaternKernel


class TestMaternKernel(unittest.TestCase):
    def test_forward_nu_1_over_2(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=0.5).initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        actual = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).div_(-lengthscale).exp()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_forward_nu_3_over_2(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=1.5).initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        dist = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).mul_(math.sqrt(3) / lengthscale)
        actual = (dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_forward_nu_5_over_2(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=2.5).initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        dist = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).mul_(math.sqrt(5) / lengthscale)
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_ard(self):
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        b = torch.tensor([[1, 4], [1, 4]], dtype=torch.float)
        lengthscales = torch.tensor([1, 2], dtype=torch.float).view(1, 1, 2)

        kernel = MaternKernel(nu=2.5, ard_num_dims=2)
        kernel.initialize(log_lengthscale=torch.log(lengthscales))
        kernel.eval()

        dist = torch.tensor([[1, 1], [2, 2]], dtype=torch.float)
        dist.mul_(math.sqrt(5))
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        dist = torch.tensor([[[0, 0], [2, 2]], [[1, 1], [0, 0]]], dtype=torch.float)
        dist.mul_(math.sqrt(5))
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b, batch_dims=(0, 2)).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, batch_dims=(0, 2)).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_ard_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 3]], [[2, -1, 2], [2, -1, 0]]], dtype=torch.float)
        b = torch.tensor([[[1, 4, 3]], [[2, -1, 0]]], dtype=torch.float)
        lengthscales = torch.tensor([[[1, 2, 1]]], dtype=torch.float)

        kernel = MaternKernel(nu=2.5, batch_size=2, ard_num_dims=3)
        kernel.initialize(log_lengthscale=torch.log(lengthscales))
        kernel.eval()

        dist = torch.tensor([[[1], [1]], [[2], [0]]], dtype=torch.float).mul_(math.sqrt(5))
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_ard_separate_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 3]], [[2, -1, 2], [2, -1, 0]]], dtype=torch.float)
        b = torch.tensor([[[1, 4, 3]], [[2, -1, 0]]], dtype=torch.float).repeat(1, 2, 1)
        lengthscales = torch.tensor([[[1, 2, 1]], [[2, 1, 0.5]]], dtype=torch.float)

        kernel = MaternKernel(nu=2.5, batch_size=2, ard_num_dims=3)
        kernel.initialize(log_lengthscale=torch.log(lengthscales))
        kernel.eval()

        dist = torch.tensor([[[1, 1], [1, 1]], [[4, 4], [0, 0]]], dtype=torch.float).mul_(math.sqrt(5))
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

        # diag
        res = kernel(a, b).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        dist = torch.tensor(
            [
                [[0, 0], [1, 1]],
                [[1, 1], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[4, 4], [0, 0]],
            ],
            dtype=torch.float,
        )
        dist.mul_(math.sqrt(5))
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b, batch_dims=(0, 2)).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, batch_dims=(0, 2)).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
