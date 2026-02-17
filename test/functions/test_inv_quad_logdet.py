#!/usr/bin/env python3

import unittest

import torch

from gpytorch.functions import TensorInvQuadLogdet
from gpytorch.kernels import RBFKernel


class TestInvQuadLogdet(unittest.TestCase):
    def test_inv_quad_logdet(self):
        # NOTE: Use small matrices here to avoid flakiness since we are testing in `float32` and `torch.allclose` by
        # default is pretty stringent.
        num_data = 3
        jitter = 1e-4

        train_x = torch.linspace(0, 1, num_data).view(num_data, 1)

        # Forward and backward using `InvQuadLogdet`
        covar_module = RBFKernel()
        covar_matrix = covar_module(train_x).evaluate_kernel().add_jitter(jitter).to_dense()

        inv_quad_rhs = torch.linspace(0, 1, num_data).requires_grad_(True)

        inv_quad, logdet = TensorInvQuadLogdet.apply(covar_matrix, inv_quad_rhs.unsqueeze(-1))
        inv_quad_logdet = inv_quad + logdet
        inv_quad_logdet.backward()

        # Forward and backward using linear operators
        covar_module_linop = RBFKernel()
        covar_matrix_linop = covar_module_linop(train_x).evaluate_kernel().add_jitter(jitter)

        inv_quad_rhs_linop = inv_quad_rhs.detach().clone().requires_grad_(True)

        inv_quad_linop, logdet_linop = covar_matrix_linop.inv_quad_logdet(inv_quad_rhs_linop.unsqueeze(-1), logdet=True)
        inv_quad_logdet_linop = inv_quad_linop + logdet_linop
        inv_quad_logdet_linop.backward()

        self.assertTrue(torch.allclose(inv_quad, inv_quad_linop))
        self.assertTrue(torch.allclose(logdet, logdet_linop))
        self.assertTrue(torch.allclose(inv_quad_logdet, inv_quad_logdet_linop))
        self.assertTrue(torch.allclose(covar_module.raw_lengthscale.grad, covar_module_linop.raw_lengthscale.grad))
        self.assertTrue(torch.allclose(inv_quad_rhs.grad, inv_quad_rhs_linop.grad))

    def test_batch_inv_quad_logdet(self):
        num_data = 3
        jitter = 1e-4

        train_x = torch.linspace(0, 1, 2 * num_data).view(2, num_data, 1)

        # Forward and backward using `InvQuadLogdet`
        covar_module = RBFKernel(batch_shape=torch.Size([2]))
        covar_matrix = covar_module(train_x).evaluate_kernel().add_jitter(jitter).to_dense()

        inv_quad_rhs = torch.linspace(0, 1, 2 * num_data).view(2, num_data).requires_grad_(True)

        inv_quad, logdet = TensorInvQuadLogdet.apply(covar_matrix, inv_quad_rhs.unsqueeze(-1))
        inv_quad_logdet = torch.sum(inv_quad + logdet)
        inv_quad_logdet.backward()

        # Forward and backward using linear operators
        covar_module_linop = RBFKernel(batch_shape=torch.Size([2]))
        covar_matrix_linop = covar_module_linop(train_x).evaluate_kernel().add_jitter(jitter)

        inv_quad_rhs_linop = inv_quad_rhs.detach().clone().requires_grad_(True)

        inv_quad_linop, logdet_linop = covar_matrix_linop.inv_quad_logdet(inv_quad_rhs_linop.unsqueeze(-1), logdet=True)
        inv_quad_logdet_linop = torch.sum(inv_quad_linop + logdet_linop)
        inv_quad_logdet_linop.backward()

        self.assertTrue(torch.allclose(inv_quad, inv_quad_linop))
        self.assertTrue(torch.allclose(logdet, logdet_linop))
        self.assertTrue(torch.allclose(inv_quad_logdet, inv_quad_logdet_linop))
        self.assertTrue(torch.allclose(covar_module.raw_lengthscale.grad, covar_module_linop.raw_lengthscale.grad))
        self.assertTrue(torch.allclose(inv_quad_rhs.grad, inv_quad_rhs_linop.grad))


if __name__ == "__main__":
    unittest.main()
