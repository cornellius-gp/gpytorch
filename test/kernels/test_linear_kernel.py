#!/usr/bin/env python3

import torch
import unittest
from gpytorch.kernels import LinearKernel


class TestLinearKernel(unittest.TestCase):
    def test_computes_linear_function_rectangular(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 1], dtype=torch.float).view(3, 1)

        kernel = LinearKernel(num_dimensions=1).initialize(offset=0, variance=1.0)
        kernel.eval()
        actual = 1 + torch.matmul(a, b.t())
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_linear_function_square(self):
        a = torch.tensor([[4, 1], [2, 0], [8, 3]], dtype=torch.float)

        offset = torch.randn(1, 1, 2)
        kernel = LinearKernel(num_dimensions=2).initialize(offset=offset, variance=1.0)
        kernel.eval()
        actual = 1 + torch.matmul(a - offset[0], (a - offset[0]).t())
        res = kernel(a, a).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, a).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        dim_group_a = a - offset[0]
        dim_group_a = dim_group_a.permute(1, 0).contiguous().view(-1, 3)
        actual = 1 + torch.mul(dim_group_a.unsqueeze(-1), dim_group_a.unsqueeze(-2))
        res = kernel(a, a, batch_dims=(0, 2)).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, a, batch_dims=(0, 2)).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_linear_function_square_batch(self):
        a = torch.tensor([[[4, 1], [2, 0], [8, 3]], [[1, 1], [2, 1], [1, 3]]], dtype=torch.float)

        offset = torch.randn(1, 1, 2)
        kernel = LinearKernel(num_dimensions=2).initialize(offset=offset, variance=1.0)
        kernel.eval()
        actual = 1 + torch.matmul(a - offset, (a - offset).transpose(-1, -2))
        res = kernel(a, a).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, a).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        dim_group_a = a - offset
        dim_group_a = dim_group_a.permute(0, 2, 1).contiguous().view(-1, 3)
        actual = 1 + torch.mul(dim_group_a.unsqueeze(-1), dim_group_a.unsqueeze(-2))
        res = kernel(a, a, batch_dims=(0, 2)).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, a, batch_dims=(0, 2)).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
