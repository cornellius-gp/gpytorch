#!/usr/bin/env python3

import torch
import unittest
from gpytorch.kernels import WhiteNoiseKernel


class TestWhiteNoiseKernel(unittest.TestCase):
    def test_computes_diag_train(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        variances = torch.randn(3)
        kernel = WhiteNoiseKernel(variances=variances)
        actual = torch.diag(variances)
        res = kernel(a).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_diag_train_batch(self):
        a = torch.tensor([[4, 2, 8], [4, 2, 8]], dtype=torch.float).view(2, 3, 1)
        variances = torch.randn(2, 3, 1)
        kernel = WhiteNoiseKernel(variances=variances)
        actual = torch.cat(
            (torch.diag(variances[0].squeeze(-1)).unsqueeze(0), torch.diag(variances[1].squeeze(-1)).unsqueeze(0))
        )
        res = kernel(a).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_zero_eval(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([3, 7], dtype=torch.float).view(2, 1)
        variances = torch.randn(3)
        kernel = WhiteNoiseKernel(variances=variances)
        kernel.eval()
        actual_one = torch.zeros(3, 2)
        actual_two = torch.zeros(2, 3)
        res_one = kernel(a, b).evaluate()
        res_two = kernel(b, a).evaluate()
        self.assertLess(torch.norm(res_one - actual_one), 1e-5)
        self.assertLess(torch.norm(res_two - actual_two), 1e-5)

    def test_computes_zero_eval_batch(self):
        a = torch.tensor([[4, 2, 8], [4, 2, 8]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[3, 7], [3, 7]], dtype=torch.float).view(2, 2, 1)
        variances = torch.randn(2, 3, 1)
        kernel = WhiteNoiseKernel(variances=variances)
        kernel.eval()
        actual_one = torch.zeros(3, 2)
        actual_two = torch.zeros(2, 3)
        res_one = kernel(a, b).evaluate()
        res_two = kernel(b, a).evaluate()
        self.assertLess(torch.norm(res_one - actual_one), 1e-5)
        self.assertLess(torch.norm(res_two - actual_two), 1e-5)

    def test_computes_diag_eval(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        variances = torch.randn(3)
        kernel = WhiteNoiseKernel(variances=variances)
        kernel.eval()
        actual = torch.diag(variances)
        res = kernel(a).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_diag_eval_batch(self):
        a = torch.tensor([[4, 2, 8], [4, 2, 8]], dtype=torch.float).view(2, 3, 1)
        variances = torch.randn(2, 3, 1)
        kernel = WhiteNoiseKernel(variances=variances)
        kernel.eval()
        actual = torch.cat(
            (torch.diag(variances[0].squeeze(-1)).unsqueeze(0), torch.diag(variances[1].squeeze(-1)).unsqueeze(0))
        )
        res = kernel(a).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
