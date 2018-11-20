#!/usr/bin/env python3

import torch
import unittest
import os
import random
from gpytorch import settings
from gpytorch.lazy import NonLazyTensor


class TestInvMatmulNonBatch(unittest.TestCase):
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

        mat = torch.randn(8, 8)
        mat = mat @ mat.transpose(-1, -2)
        vec = torch.randn(8)
        vecs = torch.randn(8, 4)
        self.mat = mat.detach().clone().requires_grad_(True)
        self.mat_copy = mat.detach().clone().requires_grad_(True)
        self.vec = vec.detach().clone().requires_grad_(True)
        self.vec_copy = vec.detach().clone().requires_grad_(True)
        self.vecs = vecs.detach().clone().requires_grad_(True)
        self.vecs_copy = vecs.detach().clone().requires_grad_(True)

    def test_inv_matmul_vec(self):
        # Forward
        with settings.terminate_cg_by_size(False):
            res = NonLazyTensor(self.mat).inv_matmul(self.vec)
            actual = self.mat_copy.inverse().matmul(self.vec_copy)
            self.assertLess(torch.max((res - actual).abs() / actual.abs()).item(), 1e-3)

            # Backward
            grad_output = torch.randn(8)
            res.backward(gradient=grad_output)
            actual.backward(gradient=grad_output)
            self.assertLess(torch.max((self.mat_copy.grad - self.mat.grad).abs()).item(), 1e-3)
            self.assertLess(torch.max((self.vec_copy.grad - self.vec.grad).abs()).item(), 1e-3)

    def test_inv_matmul_multiple_vecs(self):
        # Forward
        with settings.terminate_cg_by_size(False):
            res = NonLazyTensor(self.mat).inv_matmul(self.vecs)
            actual = self.mat_copy.inverse().matmul(self.vecs_copy)
            self.assertLess(torch.max((res - actual).abs() / actual.abs()).item(), 1e-3)

            # Backward
            grad_output = torch.randn(8, 4)
            res.backward(gradient=grad_output)
            actual.backward(gradient=grad_output)
            self.assertLess(torch.max((self.mat_copy.grad - self.mat.grad).abs()).item(), 1e-3)
            self.assertLess(torch.max((self.vecs_copy.grad - self.vecs.grad).abs()).item(), 1e-3)


class TestInvMatmulBatch(unittest.TestCase):
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

        mats = torch.randn(2, 8, 8)
        mats = mats @ mats.transpose(-1, -2)
        vecs = torch.randn(2, 8, 4)
        self.mats = mats.detach().clone().requires_grad_(True)
        self.mats_copy = mats.detach().clone().requires_grad_(True)
        self.vecs = vecs.detach().clone().requires_grad_(True)
        self.vecs_copy = vecs.detach().clone().requires_grad_(True)

    def test_inv_matmul_multiple_vecs(self):
        # Forward
        with settings.terminate_cg_by_size(False):
            res = NonLazyTensor(self.mats).inv_matmul(self.vecs)
            actual = torch.cat(
                [self.mats_copy[0].inverse().unsqueeze(0), self.mats_copy[1].inverse().unsqueeze(0)]
            ).matmul(self.vecs_copy)
            self.assertLess(torch.max((res - actual).abs()).item(), 1e-3)

            # Backward
            grad_output = torch.randn(2, 8, 4)
            res.backward(gradient=grad_output)
            actual.backward(gradient=grad_output)
            self.assertLess(torch.max((self.mats_copy.grad - self.mats.grad).abs()).item(), 1e-3)
            self.assertLess(torch.max((self.vecs_copy.grad - self.vecs.grad).abs()).item(), 1e-3)


class TestInvMatmulMultiBatch(unittest.TestCase):
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

        mats = torch.randn(3, 4, 8, 8)
        mats = mats @ mats.transpose(-1, -2)
        mats.add_(torch.eye(8).mul_(1e-1).view(1, 1, 8, 8))
        vecs = torch.randn(3, 4, 8, 2)
        self.mats = mats.detach().clone().requires_grad_(True)
        self.mats_copy = mats.detach().clone().requires_grad_(True)
        self.vecs = vecs.detach().clone().requires_grad_(True)
        self.vecs_copy = vecs.detach().clone().requires_grad_(True)

    def test_inv_matmul_multiple_vecs(self):
        # Forward
        with settings.terminate_cg_by_size(False):
            res = NonLazyTensor(self.mats).inv_matmul(self.vecs)
            flattened_mats_copy = self.mats_copy.view(-1, *self.mats.shape[-2:])
            flatened_mats_inverse = torch.cat([mat.inverse().unsqueeze(0) for mat in flattened_mats_copy])
            mats_inverse = flatened_mats_inverse.view_as(self.mats)
            actual = mats_inverse.matmul(self.vecs_copy)
            self.assertLess(torch.max((res - actual).abs()).item(), 1e-3)

            # Backward
            grad_output = torch.randn(3, 4, 8, 2)
            res.backward(gradient=grad_output)
            actual.backward(gradient=grad_output)
            self.assertLess(torch.max((self.mats_copy.grad - self.mats.grad).abs()).item(), 1e-3)
            self.assertLess(torch.max((self.vecs_copy.grad - self.vecs.grad).abs()).item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
