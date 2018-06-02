from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import unittest
import gpytorch
from torch.autograd import Variable
from gpytorch.lazy import NonLazyVariable
from gpytorch.utils import approx_equal


class TestInvQuadLogDetNonBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)

        mat = torch.Tensor([[3, -1, 0], [-1, 3, 0], [0, 0, 3]])
        vec = torch.randn(3)
        vecs = torch.randn(3, 4)

        self.mat_var = Variable(mat, requires_grad=True)
        self.vec_var = Variable(vec, requires_grad=True)
        self.vecs_var = Variable(vecs, requires_grad=True)
        self.mat_var_clone = Variable(mat, requires_grad=True)
        self.vec_var_clone = Variable(vec, requires_grad=True)
        self.vecs_var_clone = Variable(vecs, requires_grad=True)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_inv_quad_log_det_vector(self):
        # Forward pass
        actual_inv_quad = self.mat_var_clone.inverse().matmul(self.vec_var_clone).mul(self.vec_var_clone).sum()
        actual_log_det = self.mat_var_clone.det().log()
        with gpytorch.settings.num_trace_samples(1000):
            nlv = NonLazyVariable(self.mat_var)
            res_inv_quad, res_log_det = nlv.inv_quad_log_det(inv_quad_rhs=self.vec_var, log_det=True)
        self.assertAlmostEqual(res_inv_quad, actual_inv_quad, places=1)
        self.assertAlmostEqual(res_log_det.item(), actual_log_det.item(), places=1)

        # Backward
        inv_quad_grad_output = torch.Tensor([3])
        log_det_grad_output = torch.Tensor([4])
        actual_inv_quad.backward(gradient=inv_quad_grad_output)
        actual_log_det.backward(log_det_grad_output)
        res_inv_quad.backward(gradient=inv_quad_grad_output, retain_graph=True)
        res_log_det.backward(gradient=log_det_grad_output)

        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vec_var_clone.grad.data, self.vec_var.grad.data))

    def test_inv_quad_only_vector(self):
        # Forward pass
        res = NonLazyVariable(self.mat_var).inv_quad(self.vec_var)
        actual = self.mat_var_clone.inverse().matmul(self.vec_var_clone).mul(self.vec_var_clone).sum()
        self.assertAlmostEqual(res.item(), actual.item(), places=1)

        # Backward
        inv_quad_grad_output = torch.randn(1)
        actual.backward(gradient=inv_quad_grad_output)
        res.backward(gradient=inv_quad_grad_output)

        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vec_var_clone.grad.data, self.vec_var.grad.data))

    def test_inv_quad_log_det_many_vectors(self):
        # Forward pass
        actual_inv_quad = self.mat_var_clone.inverse().matmul(self.vecs_var_clone).mul(self.vecs_var_clone).sum()
        actual_log_det = self.mat_var_clone.det().log()
        with gpytorch.settings.num_trace_samples(1000):
            nlv = NonLazyVariable(self.mat_var)
            res_inv_quad, res_log_det = nlv.inv_quad_log_det(inv_quad_rhs=self.vecs_var, log_det=True)
        self.assertAlmostEqual(res_inv_quad.item(), actual_inv_quad.item(), places=1)
        self.assertAlmostEqual(res_log_det.item(), actual_log_det.item(), places=1)

        # Backward
        inv_quad_grad_output = torch.Tensor([3])
        log_det_grad_output = torch.Tensor([4])
        actual_inv_quad.backward(gradient=inv_quad_grad_output)
        actual_log_det.backward(log_det_grad_output)
        res_inv_quad.backward(gradient=inv_quad_grad_output, retain_graph=True)
        res_log_det.backward(gradient=log_det_grad_output)

        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))

    def test_inv_quad_only_many_vectors(self):
        # Forward pass
        res = NonLazyVariable(self.mat_var).inv_quad(self.vecs_var)
        actual = self.mat_var_clone.inverse().matmul(self.vecs_var_clone).mul(self.vecs_var_clone).sum()
        self.assertAlmostEqual(res.item(), actual.item(), places=1)

        # Backward
        inv_quad_grad_output = torch.randn(1)
        actual.backward(gradient=inv_quad_grad_output)
        res.backward(gradient=inv_quad_grad_output)

        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))

    def test_log_det_only(self):
        # Forward pass
        with gpytorch.settings.num_trace_samples(1000):
            res = NonLazyVariable(self.mat_var).log_det()
        actual = self.mat_var_clone.det().log()
        self.assertAlmostEqual(res.item(), actual.item(), places=1)

        # Backward
        grad_output = torch.Tensor([3])
        actual.backward(gradient=grad_output)
        res.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data, epsilon=1e-1))


class TestInvQuadLogDetBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)

        mats = torch.Tensor([[[3, -1, 0], [-1, 3, 0], [0, 0, 3]], [[10, -2, 1], [-2, 10, 0], [1, 0, 10]]])
        vecs = torch.randn(2, 3, 4)

        self.mats_var = Variable(mats, requires_grad=True)
        self.vecs_var = Variable(vecs, requires_grad=True)
        self.mats_var_clone = Variable(mats, requires_grad=True)
        self.vecs_var_clone = Variable(vecs, requires_grad=True)

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
            nlv = NonLazyVariable(self.mats_var)
            res_inv_quad, res_log_det = nlv.inv_quad_log_det(inv_quad_rhs=self.vecs_var, log_det=True)
        self.assertTrue(approx_equal(res_inv_quad.data, actual_inv_quad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(res_log_det.data, actual_log_det.data, epsilon=1e-1))

        # Backward
        inv_quad_grad_output = torch.Tensor([3, 4])
        log_det_grad_output = torch.Tensor([4, 2])
        actual_inv_quad.backward(gradient=inv_quad_grad_output)
        actual_log_det.backward(gradient=log_det_grad_output)
        res_inv_quad.backward(gradient=inv_quad_grad_output, retain_graph=True)
        res_log_det.backward(gradient=log_det_grad_output)

        self.assertTrue(approx_equal(self.mats_var_clone.grad.data, self.mats_var.grad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))

    def test_inv_quad_only_many_vectors(self):
        # Forward pass
        res = NonLazyVariable(self.mats_var).inv_quad(self.vecs_var)
        actual = (
            torch.cat([self.mats_var_clone[0].inverse().unsqueeze(0), self.mats_var_clone[1].inverse().unsqueeze(0)])
            .matmul(self.vecs_var_clone)
            .mul(self.vecs_var_clone)
            .sum(2)
            .sum(1)
        )
        self.assertTrue(approx_equal(res.data, actual.data, epsilon=1e-1))

        # Backward
        inv_quad_grad_output = torch.randn(2)
        actual.backward(gradient=inv_quad_grad_output)
        res.backward(gradient=inv_quad_grad_output)

        self.assertTrue(approx_equal(self.mats_var_clone.grad.data, self.mats_var.grad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))

    def test_log_det_only(self):
        # Forward pass
        with gpytorch.settings.num_trace_samples(1000):
            res = NonLazyVariable(self.mats_var).log_det()
        actual = torch.cat(
            [self.mats_var_clone[0].det().log().unsqueeze(0), self.mats_var_clone[1].det().log().unsqueeze(0)]
        )
        self.assertTrue(approx_equal(res.data, actual.data, epsilon=1e-1))

        # Backward
        grad_output = torch.Tensor([3, 4])
        actual.backward(gradient=grad_output)
        res.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mats_var_clone.grad.data, self.mats_var.grad.data, epsilon=1e-1))


if __name__ == "__main__":
    unittest.main()
