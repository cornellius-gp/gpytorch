#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import CosineKernel, PeriodicKernel


class TestPeriodicDoubleBackward(unittest.TestCase):
    def test_second_derivatives(self):
        for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            for cls in [PeriodicKernel, CosineKernel]:
                with self.subTest(device=device, kernel=cls.__name__):
                    kernel = cls().to(device=device, dtype=torch.double)
                    x1 = torch.tensor([[0.1, 0.2], [0.7, 0.6]], device=device, dtype=torch.double, requires_grad=True)
                    x2 = x1.detach().clone().requires_grad_()
                    call = lambda a, b: kernel(a, b).to_dense()
                    self.assertTrue(torch.autograd.gradcheck(call, (x1, x2)))
                    self.assertTrue(torch.autograd.gradgradcheck(call, (x1, x2)))
                    variables = (x1, x2, *kernel.parameters())
                    gradients = torch.autograd.grad(call(x1, x2).sum(), variables, create_graph=True)
                    second = torch.autograd.grad(sum(g.square().sum() for g in gradients), variables)
                    self.assertTrue(all(g.isfinite().all() for g in second))
