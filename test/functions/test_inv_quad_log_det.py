from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import unittest
import gpytorch
from gpytorch.lazy import NonLazyTensor
from test._utils import approx_equal


class TestInvQuadLogDetNonBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)

        self.mat_var = torch.tensor([[3, -1, 0], [-1, 3, 0], [0, 0, 3]], dtype=torch.float)
        self.mat_var_clone = self.mat_var.clone()
        self.vec_var = torch.randn(3)
        self.vec_var_clone = self.vec_var.clone()
        self.vecs_var = torch.randn(3, 4)
        self.vecs_var_clone = self.vecs_var.clone()
        self.mat_var.requires_grad = True
        self.mat_var_clone.requires_grad = True
        self.vec_var.requires_grad = True
        self.vec_var_clone.requires_grad = True
        self.vecs_var.requires_grad = True
        self.vecs_var_clone.requires_grad = True

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_inv_quad_log_det_vector(self):
        # Forward pass
        actual_inv_quad = self.mat_var_clone.inverse().matmul(self.vec_var_clone).mul(self.vec_var_clone).sum()
        actual_log_det = self.mat_var_clone.det().log()
        with gpytorch.settings.num_trace_samples(1000):
            nlv = NonLazyTensor(self.mat_var)
            res_inv_quad, res_log_det = nlv.inv_quad_log_det(inv_quad_rhs=self.vec_var, log_det=True)
        self.assertAlmostEqual(res_inv_quad, actual_inv_quad, places=1)
        self.assertAlmostEqual(res_log_det.item(), actual_log_det.item(), places=1)

        # Backward
        actual_inv_quad.backward()
        actual_log_det.backward()
        res_inv_quad.backward(retain_graph=True)
        res_log_det.backward()

        self.assertTrue(approx_equal(self.mat_var_clone.grad, self.mat_var.grad, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vec_var_clone.grad, self.vec_var.grad))

    def test_inv_quad_only_vector(self):
        # Forward pass
        res = NonLazyTensor(self.mat_var).inv_quad(self.vec_var)
        actual = self.mat_var_clone.inverse().matmul(self.vec_var_clone).mul(self.vec_var_clone).sum()
        self.assertAlmostEqual(res.item(), actual.item(), places=1)

        # Backward
        actual.backward()
        res.backward()

        self.assertTrue(approx_equal(self.mat_var_clone.grad, self.mat_var.grad, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vec_var_clone.grad, self.vec_var.grad))

    def test_inv_quad_log_det_many_vectors(self):
        # Forward pass
        actual_inv_quad = self.mat_var_clone.inverse().matmul(self.vecs_var_clone).mul(self.vecs_var_clone).sum()
        actual_log_det = self.mat_var_clone.det().log()
        with gpytorch.settings.num_trace_samples(1000):
            nlv = NonLazyTensor(self.mat_var)
            res_inv_quad, res_log_det = nlv.inv_quad_log_det(inv_quad_rhs=self.vecs_var, log_det=True)
        self.assertAlmostEqual(res_inv_quad.item(), actual_inv_quad.item(), places=1)
        self.assertAlmostEqual(res_log_det.item(), actual_log_det.item(), places=1)

        # Backward
        actual_inv_quad.backward()
        actual_log_det.backward()
        res_inv_quad.backward(retain_graph=True)
        res_log_det.backward()

        self.assertTrue(approx_equal(self.mat_var_clone.grad, self.mat_var.grad, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad, self.vecs_var.grad))

    def test_inv_quad_only_many_vectors(self):
        # Forward pass
        res = NonLazyTensor(self.mat_var).inv_quad(self.vecs_var)
        actual = self.mat_var_clone.inverse().matmul(self.vecs_var_clone).mul(self.vecs_var_clone).sum()
        self.assertAlmostEqual(res.item(), actual.item(), places=1)

        # Backward
        actual.backward()
        res.backward()

        self.assertTrue(approx_equal(self.mat_var_clone.grad, self.mat_var.grad, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad, self.vecs_var.grad))

    def test_log_det_only(self):
        # Forward pass
        with gpytorch.settings.num_trace_samples(1000):
            res = NonLazyTensor(self.mat_var).log_det()
        actual = self.mat_var_clone.det().log()
        self.assertAlmostEqual(res.item(), actual.item(), places=1)

        # Backward
        actual.backward()
        res.backward()
        self.assertTrue(approx_equal(self.mat_var_clone.grad, self.mat_var.grad, epsilon=1e-1))


class TestInvQuadLogDetBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)

        mats = [[[3, -1, 0], [-1, 3, 0], [0, 0, 3]], [[10, -2, 1], [-2, 10, 0], [1, 0, 10]]]
        self.mats_var = torch.tensor(mats, dtype=torch.float, requires_grad=True)
        self.mats_var_clone = self.mats_var.clone().detach().requires_grad_(True)
        self.vecs_var = torch.randn(2, 3, 4, requires_grad=True)
        self.vecs_var_clone = self.vecs_var.clone().detach().requires_grad_(True)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_inv_quad_log_det_many_vectors(self):
        # Forward pass
        actual_inv_quad = (
            torch.cat([self.mats_var_clone[0].inverse().unsqueeze(0), self.mats_var_clone[1].inverse().unsqueeze(0)])
            .matmul(self.vecs_var_clone)
            .mul(self.vecs_var_clone)
            .sum(2)
            .sum(1)
        )
        actual_log_det = torch.cat(
            [self.mats_var_clone[0].det().log().unsqueeze(0), self.mats_var_clone[1].det().log().unsqueeze(0)]
        )
        with gpytorch.settings.num_trace_samples(1000):
            nlv = NonLazyTensor(self.mats_var)
            res_inv_quad, res_log_det = nlv.inv_quad_log_det(inv_quad_rhs=self.vecs_var, log_det=True)
        self.assertTrue(approx_equal(res_inv_quad, actual_inv_quad, epsilon=1e-1))
        self.assertTrue(approx_equal(res_log_det, actual_log_det, epsilon=1e-1))

        # Backward
        inv_quad_grad_output = torch.tensor([3, 4], dtype=torch.float)
        log_det_grad_output = torch.tensor([4, 2], dtype=torch.float)
        actual_inv_quad.backward(gradient=inv_quad_grad_output)
        actual_log_det.backward(gradient=log_det_grad_output)
        res_inv_quad.backward(gradient=inv_quad_grad_output, retain_graph=True)
        res_log_det.backward(gradient=log_det_grad_output)

        self.assertTrue(approx_equal(self.mats_var_clone.grad, self.mats_var.grad, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad, self.vecs_var.grad))

    def test_inv_quad_only_many_vectors(self):
        # Forward pass
        res = NonLazyTensor(self.mats_var).inv_quad(self.vecs_var).sum()
        actual = (
            torch.cat([self.mats_var_clone[0].inverse().unsqueeze(0), self.mats_var_clone[1].inverse().unsqueeze(0)])
            .matmul(self.vecs_var_clone)
            .mul(self.vecs_var_clone)
            .sum(2)
            .sum(1)
        ).sum()
        self.assertTrue(approx_equal(res, actual, epsilon=1e-1))
        # Backward
        actual.backward()
        res.backward()

        self.assertTrue(approx_equal(self.mats_var_clone.grad, self.mats_var.grad, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad, self.vecs_var.grad))

    def test_log_det_only(self):
        # Forward pass
        with gpytorch.settings.num_trace_samples(1000):
            res = NonLazyTensor(self.mats_var).log_det()
        actual = torch.cat(
            [self.mats_var_clone[0].det().log().unsqueeze(0), self.mats_var_clone[1].det().log().unsqueeze(0)]
        )
        self.assertTrue(approx_equal(res, actual, epsilon=1e-1))

        # Backward
        grad_output = torch.tensor([3, 4], dtype=torch.float)
        actual.backward(gradient=grad_output)
        res.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mats_var_clone.grad, self.mats_var.grad, epsilon=1e-1))


if __name__ == "__main__":
    unittest.main()
