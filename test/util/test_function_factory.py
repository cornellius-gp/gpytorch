import os
import math
import torch
import unittest
import gpytorch
import numpy as np
from torch.autograd import Variable
from gpytorch.utils import approx_equal
from gpytorch.lazy import NonLazyVariable


class TestMatmulNonBatch(unittest.TestCase):
    def setUp(self):
        mat = torch.Tensor([
            [3, -1, 0],
            [-1, 3, 0],
            [0, 0, 3],
        ])
        vec = torch.randn(3)
        vecs = torch.randn(3, 4)

        self.mat_var = Variable(mat, requires_grad=True)
        self.mat_var_clone = Variable(mat, requires_grad=True)
        self.vec_var = Variable(vec, requires_grad=True)
        self.vec_var_clone = Variable(vec, requires_grad=True)
        self.vecs_var = Variable(vecs, requires_grad=True)
        self.vecs_var_clone = Variable(vecs, requires_grad=True)

    def test_matmul_vec(self):
        # Forward
        res = NonLazyVariable(self.mat_var).matmul(self.vec_var)
        actual = self.mat_var_clone.matmul(self.vec_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.Tensor(3)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data))
        self.assertTrue(approx_equal(self.vec_var_clone.grad.data, self.vec_var.grad.data))

    def test_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyVariable(self.mat_var).matmul(self.vecs_var)
        actual = self.mat_var_clone.matmul(self.vecs_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.Tensor(3, 4)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))


class TestMatmulBatch(unittest.TestCase):
    def setUp(self):
        mats = torch.Tensor([
            [
                [3, -1, 0],
                [-1, 3, 0],
                [0, 0, 3],
            ], [
                [10, -2, 1],
                [-2, 10, 0],
                [1, 0, 10],
            ]
        ])
        vecs = torch.randn(2, 3, 4)

        self.mats_var = Variable(mats, requires_grad=True)
        self.mats_var_clone = Variable(mats, requires_grad=True)
        self.vecs_var = Variable(vecs, requires_grad=True)
        self.vecs_var_clone = Variable(vecs, requires_grad=True)

    def test_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyVariable(self.mats_var).matmul(self.vecs_var)
        actual = self.mats_var_clone.matmul(self.vecs_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.Tensor(2, 3, 4)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mats_var_clone.grad.data, self.mats_var.grad.data))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))


class TestInvMatmulNonBatch(unittest.TestCase):
    def setUp(self):
        mat = torch.Tensor([
            [3, -1, 0],
            [-1, 3, 0],
            [0, 0, 3],
        ])
        vec = torch.randn(3)
        vecs = torch.randn(3, 4)

        self.mat_var = Variable(mat, requires_grad=True)
        self.mat_var_clone = Variable(mat, requires_grad=True)
        self.vec_var = Variable(vec, requires_grad=True)
        self.vec_var_clone = Variable(vec, requires_grad=True)
        self.vecs_var = Variable(vecs, requires_grad=True)
        self.vecs_var_clone = Variable(vecs, requires_grad=True)

    def test_inv_matmul_vec(self):
        # Forward
        res = NonLazyVariable(self.mat_var).inv_matmul(self.vec_var)
        actual = self.mat_var_clone.inverse().matmul(self.vec_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(3)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data))
        self.assertTrue(approx_equal(self.vec_var_clone.grad.data, self.vec_var.grad.data))

    def test_inv_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyVariable(self.mat_var).inv_matmul(self.vecs_var)
        actual = self.mat_var_clone.inverse().matmul(self.vecs_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(3, 4)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))


class TestInvMatmulBatch(unittest.TestCase):
    def setUp(self):
        mats = torch.Tensor([
            [
                [3, -1, 0],
                [-1, 3, 0],
                [0, 0, 3],
            ], [
                [10, -2, 1],
                [-2, 10, 0],
                [1, 0, 10],
            ]
        ])
        vecs = torch.randn(2, 3, 4)

        self.mats_var = Variable(mats, requires_grad=True)
        self.mats_var_clone = Variable(mats, requires_grad=True)
        self.vecs_var = Variable(vecs, requires_grad=True)
        self.vecs_var_clone = Variable(vecs, requires_grad=True)

    def test_inv_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyVariable(self.mats_var).inv_matmul(self.vecs_var)
        actual = torch.cat([
            self.mats_var_clone[0].inverse().unsqueeze(0),
            self.mats_var_clone[1].inverse().unsqueeze(0),
        ]).matmul(self.vecs_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(2, 3, 4)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mats_var_clone.grad.data, self.mats_var.grad.data))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))


class TestInvQuadLogDetNonBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv('UNLOCK_SEED') is None or os.getenv('UNLOCK_SEED').lower() == 'false':
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)

        mat = torch.Tensor([
            [3, -1, 0],
            [-1, 3, 0],
            [0, 0, 3],
        ])
        vec = torch.randn(3)
        vecs = torch.randn(3, 4)

        self.mat_var = Variable(mat, requires_grad=True)
        self.vec_var = Variable(vec, requires_grad=True)
        self.vecs_var = Variable(vecs, requires_grad=True)
        self.mat_var_clone = Variable(mat, requires_grad=True)
        self.vec_var_clone = Variable(vec, requires_grad=True)
        self.vecs_var_clone = Variable(vecs, requires_grad=True)
        self.log_det = math.log(np.linalg.det(mat.numpy()))

    def tearDown(self):
        if hasattr(self, 'rng_state'):
            torch.set_rng_state(self.rng_state)

    def test_inv_quad_log_det_vector(self):
        # Forward pass
        actual_inv_quad = self.mat_var_clone.inverse().matmul(self.vec_var_clone).dot(self.vec_var_clone)
        with gpytorch.settings.num_trace_samples(1000):
            nlv = NonLazyVariable(self.mat_var)
            res_inv_quad, res_log_det = nlv.inv_quad_log_det(inv_quad_rhs=self.vec_var, log_det=True)
        self.assertAlmostEqual(res_inv_quad.data[0], actual_inv_quad.data[0], places=1)
        self.assertAlmostEqual(res_log_det.data[0], self.log_det, places=1)

        # Backward
        inv_quad_grad_output = torch.Tensor([3])
        log_det_grad_output = torch.Tensor([4])
        actual_inv_quad.backward(gradient=inv_quad_grad_output)
        self.mat_var_clone.grad.data.add_(self.mat_var_clone.data.inverse() * log_det_grad_output)
        res_inv_quad.backward(gradient=inv_quad_grad_output, retain_graph=True)
        res_log_det.backward(gradient=log_det_grad_output)

        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vec_var_clone.grad.data, self.vec_var.grad.data))

    def test_inv_quad_only_vector(self):
        # Forward pass
        res = NonLazyVariable(self.mat_var).inv_quad(self.vec_var)
        actual = self.mat_var_clone.inverse().matmul(self.vec_var_clone).mul(self.vec_var_clone).sum()
        self.assertAlmostEqual(res.data[0], actual.data[0], places=1)

        # Backward
        inv_quad_grad_output = torch.randn(1)
        actual.backward(gradient=inv_quad_grad_output)
        res.backward(gradient=inv_quad_grad_output)

        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vec_var_clone.grad.data, self.vec_var.grad.data))

    def test_inv_quad_log_det_many_vectors(self):
        # Forward pass
        actual_inv_quad = self.mat_var_clone.inverse().matmul(self.vecs_var_clone).dot(self.vecs_var_clone)
        with gpytorch.settings.num_trace_samples(1000):
            nlv = NonLazyVariable(self.mat_var)
            res_inv_quad, res_log_det = nlv.inv_quad_log_det(inv_quad_rhs=self.vecs_var, log_det=True)
        self.assertAlmostEqual(res_inv_quad.data[0], actual_inv_quad.data[0], places=1)
        self.assertAlmostEqual(res_log_det.data[0], self.log_det, places=1)

        # Backward
        inv_quad_grad_output = torch.Tensor([3])
        log_det_grad_output = torch.Tensor([4])
        actual_inv_quad.backward(gradient=inv_quad_grad_output)
        self.mat_var_clone.grad.data.add_(self.mat_var_clone.data.inverse() * log_det_grad_output)
        res_inv_quad.backward(gradient=inv_quad_grad_output, retain_graph=True)
        res_log_det.backward(gradient=log_det_grad_output)

        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))

    def test_inv_quad_only_many_vectors(self):
        # Forward pass
        res = NonLazyVariable(self.mat_var).inv_quad(self.vecs_var)
        actual = self.mat_var_clone.inverse().matmul(self.vecs_var_clone).mul(self.vecs_var_clone).sum()
        self.assertAlmostEqual(res.data[0], actual.data[0], places=1)

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
        self.assertAlmostEqual(res.data[0], self.log_det, places=1)

        # Backward
        grad_output = torch.Tensor([3])
        actual_mat_grad = self.mat_var_clone.data.inverse().mul(grad_output)
        res.backward(gradient=grad_output)
        self.assertTrue(approx_equal(actual_mat_grad, self.mat_var.grad.data, epsilon=1e-1))


class TestInvQuadLogDetBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv('UNLOCK_SEED') is None or os.getenv('UNLOCK_SEED').lower() == 'false':
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)

        mats = torch.Tensor([
            [
                [3, -1, 0],
                [-1, 3, 0],
                [0, 0, 3],
            ], [
                [10, -2, 1],
                [-2, 10, 0],
                [1, 0, 10],
            ]
        ])
        vecs = torch.randn(2, 3, 4)

        self.mats_var = Variable(mats, requires_grad=True)
        self.vecs_var = Variable(vecs, requires_grad=True)
        self.mats_var_clone = Variable(mats, requires_grad=True)
        self.vecs_var_clone = Variable(vecs, requires_grad=True)
        self.log_dets = torch.Tensor([
            math.log(np.linalg.det(mats[0].numpy())),
            math.log(np.linalg.det(mats[1].numpy())),
        ])

    def tearDown(self):
        if hasattr(self, 'rng_state'):
            torch.set_rng_state(self.rng_state)

    def test_inv_quad_log_det_many_vectors(self):
        # Forward pass
        actual_inv_quad = torch.cat([
            self.mats_var_clone[0].inverse().unsqueeze(0),
            self.mats_var_clone[1].inverse().unsqueeze(0),
        ]).matmul(self.vecs_var_clone).mul(self.vecs_var_clone).sum(2).sum(1)
        with gpytorch.settings.num_trace_samples(1000):
            nlv = NonLazyVariable(self.mats_var)
            res_inv_quad, res_log_det = nlv.inv_quad_log_det(inv_quad_rhs=self.vecs_var, log_det=True)
        for i in range(self.mats_var.size(0)):
            self.assertAlmostEqual(res_inv_quad.data[i], actual_inv_quad.data[i], places=1)
            self.assertAlmostEqual(res_log_det.data[i], self.log_dets[i], places=1)

        # Backward
        inv_quad_grad_output = torch.Tensor([3, 4])
        log_det_grad_output = torch.Tensor([4, 2])
        actual_inv_quad.backward(gradient=inv_quad_grad_output)
        mat_log_det_grad = torch.cat([
            self.mats_var_clone[0].data.inverse().mul(log_det_grad_output[0]).unsqueeze(0),
            self.mats_var_clone[1].data.inverse().mul(log_det_grad_output[1]).unsqueeze(0),
        ])
        self.mats_var_clone.grad.data.add_(mat_log_det_grad)
        res_inv_quad.backward(gradient=inv_quad_grad_output, retain_graph=True)
        res_log_det.backward(gradient=log_det_grad_output)

        self.assertTrue(approx_equal(self.mats_var_clone.grad.data, self.mats_var.grad.data, epsilon=1e-1))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))

    def test_inv_quad_only_many_vectors(self):
        # Forward pass
        res = NonLazyVariable(self.mats_var).inv_quad(self.vecs_var)
        actual = torch.cat([
            self.mats_var_clone[0].inverse().unsqueeze(0),
            self.mats_var_clone[1].inverse().unsqueeze(0),
        ]).matmul(self.vecs_var_clone).mul(self.vecs_var_clone).sum(2).sum(1)
        for i in range(self.mats_var.size(0)):
            self.assertAlmostEqual(res.data[i], actual.data[i], places=1)

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
        for i in range(self.mats_var.size(0)):
            self.assertAlmostEqual(res.data[i], self.log_dets[i], places=1)

        # Backward
        grad_output = torch.Tensor([3, 4])
        actual_mat_grad = torch.cat([
            self.mats_var_clone[0].data.inverse().mul(grad_output[0]).unsqueeze(0),
            self.mats_var_clone[1].data.inverse().mul(grad_output[1]).unsqueeze(0),
        ])
        res.backward(gradient=grad_output)
        self.assertTrue(approx_equal(actual_mat_grad, self.mats_var.grad.data, epsilon=1e-1))


class TestRootDecomposition(unittest.TestCase):
    def setUp(self):
        if os.getenv('UNLOCK_SEED') is None or os.getenv('UNLOCK_SEED').lower() == 'false':
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

        mat = torch.Tensor([
            [5.0212, 0.5504, -0.1810, 1.5414, 2.9611],
            [0.5504, 2.8000, 1.9944, 0.6208, -0.8902],
            [-0.1810, 1.9944, 3.0505, 1.0790, -1.1774],
            [1.5414, 0.6208, 1.0790, 2.9430, 0.4170],
            [2.9611, -0.8902, -1.1774, 0.4170, 3.3208],
        ])
        self.mat_var = Variable(mat, requires_grad=True)
        self.mat_var_clone = Variable(mat, requires_grad=True)

    def tearDown(self):
        if hasattr(self, 'rng_state'):
            torch.set_rng_state(self.rng_state)

    def test_root_decomposition(self):
        # Forward
        root = NonLazyVariable(self.mat_var).root_decomposition()
        res = root.matmul(root.transpose(-1, -2))
        self.assertTrue(approx_equal(res.data, self.mat_var.data))

        # Backward
        res.trace().backward()
        self.mat_var_clone.trace().backward()
        self.assertTrue(approx_equal(self.mat_var.grad.data, self.mat_var_clone.grad.data))

    def test_root_inv_decomposition(self):
        # Forward
        root = NonLazyVariable(self.mat_var).root_inv_decomposition()
        res = root.matmul(root.transpose(-1, -2))
        actual = self.mat_var_clone.inverse()
        self.assertTrue(approx_equal(res.data, actual.data))

        # Backward
        res.trace().backward()
        actual.trace().backward()
        self.assertTrue(approx_equal(self.mat_var.grad.data, self.mat_var_clone.grad.data))


if __name__ == '__main__':
    unittest.main()
