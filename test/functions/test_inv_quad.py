#!/usr/bin/env python3

import os
import random
import unittest

import torch

import gpytorch
from gpytorch.lazy import NonLazyTensor


class TestInvQuadNonBatch(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def setUp(self):
        if os.getenv("unlock_seed") is None or os.getenv("unlock_seed").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

        mat = torch.randn(4, 4)
        mat = mat @ mat.transpose(-1, -2)
        mat.div_(5).add_(torch.eye(4))
        vecs = torch.randn(5, 4, 6)
        vec = torch.randn(4)
        vecs = torch.randn(4, 8)
        self.mat = mat.detach().clone().requires_grad_(True)
        self.mat_clone = mat.detach().clone().requires_grad_(True)
        self.vec = vec.detach().clone().requires_grad_(True)
        self.vec_clone = vec.detach().clone().requires_grad_(True)
        self.vecs = vecs.detach().clone().requires_grad_(True)
        self.vecs_clone = vecs.detach().clone().requires_grad_(True)

    def test_inv_quad_vector(self):
        # Forward pass
        actual_inv_quad = self.mat_clone.inverse().matmul(self.vec_clone).mul(self.vec_clone).sum()
        with gpytorch.settings.num_trace_samples(1000):
            non_lazy_tsr = NonLazyTensor(self.mat)
            res_inv_quad = non_lazy_tsr.inv_quad(self.vec)

        self.assertAlmostEqual(res_inv_quad.item(), actual_inv_quad.item(), places=1)

        # Backward
        actual_inv_quad.backward()
        res_inv_quad.backward(retain_graph=True)

        self.assertLess(torch.max((self.mat_clone.grad - self.mat.grad).abs()).item(), 1e-1)
        self.assertLess(torch.max((self.vec_clone.grad - self.vec.grad).abs()).item(), 1e-1)

    def test_inv_quad_many_vectors(self):
        # Forward pass
        actual_inv_quad = self.mat_clone.inverse().matmul(self.vecs_clone).mul(self.vecs_clone).sum()
        with gpytorch.settings.num_trace_samples(1000):
            non_lazy_tsr = NonLazyTensor(self.mat)
            res_inv_quad = non_lazy_tsr.inv_quad(self.vecs)
        self.assertAlmostEqual(res_inv_quad.item(), actual_inv_quad.item(), places=1)

        # Backward
        actual_inv_quad.backward()
        res_inv_quad.backward(retain_graph=True)

        self.assertLess(torch.max((self.mat_clone.grad - self.mat.grad).abs()).item(), 1e-1)
        self.assertLess(torch.max((self.vecs_clone.grad - self.vecs.grad).abs()).item(), 1e-1)


class TestInvQuadBatch(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def setUp(self):
        if os.getenv("unlock_seed") is None or os.getenv("unlock_seed").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1)
            random.seed(1)

        mats = torch.randn(5, 4, 4)
        mats = mats @ mats.transpose(-1, -2)
        mats.div_(5).add_(torch.eye(4).unsqueeze_(0))
        vecs = torch.randn(5, 4, 6)
        self.mats = mats.detach().clone().requires_grad_(True)
        self.mats_clone = mats.detach().clone().requires_grad_(True)
        self.vecs = vecs.detach().clone().requires_grad_(True)
        self.vecs_clone = vecs.detach().clone().requires_grad_(True)

    def test_inv_quad_many_vectors(self):
        # Forward pass
        actual_inv_quad = (
            torch.cat([mat.inverse().unsqueeze(0) for mat in self.mats_clone])
            .matmul(self.vecs_clone)
            .mul(self.vecs_clone)
            .sum(2)
            .sum(1)
        )
        with gpytorch.settings.num_trace_samples(2000):
            non_lazy_tsr = NonLazyTensor(self.mats)
            res_inv_quad = non_lazy_tsr.inv_quad(self.vecs)

        self.assertEqual(res_inv_quad.shape, actual_inv_quad.shape)
        self.assertLess(torch.max((res_inv_quad - actual_inv_quad).abs()).item(), 1e-1)

        # Backward
        inv_quad_grad_output = torch.randn(5, dtype=torch.float)
        actual_inv_quad.backward(gradient=inv_quad_grad_output)
        res_inv_quad.backward(gradient=inv_quad_grad_output, retain_graph=True)

        self.assertLess(torch.max((self.mats_clone.grad - self.mats.grad).abs()).item(), 1e-1)
        self.assertLess(torch.max((self.vecs_clone.grad - self.vecs.grad).abs()).item(), 1e-1)


class TestInvQuadMultiBatch(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def setUp(self):
        if os.getenv("unlock_seed") is None or os.getenv("unlock_seed").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

        mats = torch.randn(2, 3, 4, 4)
        mats = mats @ mats.transpose(-1, -2)
        mats.div_(5).add_(torch.eye(4).view(1, 1, 4, 4))
        vecs = torch.randn(2, 3, 4, 6)
        self.mats = mats.detach().clone().requires_grad_(True)
        self.mats_clone = mats.detach().clone().requires_grad_(True)
        self.vecs = vecs.detach().clone().requires_grad_(True)
        self.vecs_clone = vecs.detach().clone().requires_grad_(True)

    def test_inv_quad_many_vectors(self):
        # Forward pass
        flattened_mats = self.mats_clone.view(-1, *self.mats_clone.shape[-2:])
        actual_inv_quad = (
            torch.cat([mat.inverse().unsqueeze(0) for mat in flattened_mats])
            .view(self.mats_clone.shape)
            .matmul(self.vecs_clone)
            .mul(self.vecs_clone)
            .sum(-2)
            .sum(-1)
        )

        with gpytorch.settings.num_trace_samples(2000):
            non_lazy_tsr = NonLazyTensor(self.mats)
            res_inv_quad = non_lazy_tsr.inv_quad(self.vecs)

        self.assertEqual(res_inv_quad.shape, actual_inv_quad.shape)
        self.assertLess(torch.max((res_inv_quad - actual_inv_quad).abs()).item(), 1e-1)

        # Backward
        inv_quad_grad_output = torch.randn(2, 3, dtype=torch.float)
        actual_inv_quad.backward(gradient=inv_quad_grad_output)
        res_inv_quad.backward(gradient=inv_quad_grad_output, retain_graph=True)

        self.assertLess(torch.max((self.mats_clone.grad - self.mats.grad).abs()).item(), 1e-1)
        self.assertLess(torch.max((self.vecs_clone.grad - self.vecs.grad).abs()).item(), 1e-1)


if __name__ == "__main__":
    unittest.main()
