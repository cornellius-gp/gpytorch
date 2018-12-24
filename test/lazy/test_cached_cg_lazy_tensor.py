#!/usr/bin/env python3

import math
import torch
import gpytorch
import unittest
import warnings
from gpytorch.lazy import CachedCGLazyTensor, NonLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


class TestCachedCGLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        mat = torch.randn(5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)

        lazy_tensor = NonLazyTensor(mat)
        eager_rhs = torch.randn(5, 10).detach()
        with gpytorch.settings.num_trace_samples(1000):  # For inv_quad_log_det tests
            solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLazyTensor.precompute_terms(
                lazy_tensor, eager_rhs.detach()
            )
            eager_rhss = [eager_rhs.detach(), eager_rhs[..., -2:-1].detach()]
            solves = [solve.detach(), solve[..., -2:-1].detach()]

        return CachedCGLazyTensor(
            lazy_tensor, eager_rhss, solves, probe_vecs, probe_vec_norms, probe_vec_solves, tmats
        )

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.base_lazy_tensor.tensor

    def test_inv_matmul_vec(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = lazy_tensor.eager_rhss[1].squeeze(-1).clone().detach().requires_grad_(True)
        test_vector_copy = lazy_tensor_copy.eager_rhss[1].squeeze(-1).clone().detach().requires_grad_(True)
        # Make sure that we get no warning about CG
        with gpytorch.settings.max_cg_iterations(200), warnings.catch_warnings(record=True) as w:
            res = lazy_tensor.inv_matmul(test_vector)
            actual = evaluated.inverse().matmul(test_vector_copy)
            self.assertEqual(len(w), 0)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        # Make sure that we get a warning that CG was run
        with warnings.catch_warnings(record=True) as w:
            res.backward(gradient=grad)
            actual.backward(gradient=grad)
            self.assertEqual(len(w), 1)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )
        self.assertLess(
            ((test_vector.grad - test_vector_copy.grad).abs() / test_vector.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )

    def test_inv_matmul_vector_with_left(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = lazy_tensor.eager_rhss[0][..., -1].squeeze(-1).clone().detach().requires_grad_(True)
        test_vector_copy = lazy_tensor_copy.eager_rhss[0][..., -1].squeeze(-1).clone().detach().requires_grad_(True)
        test_left = lazy_tensor.eager_rhss[0][..., :-1].t().clone().detach().requires_grad_(True)
        test_left_copy = lazy_tensor_copy.eager_rhss[0][..., :-1].t().clone().detach().requires_grad_(True)
        # Make sure that we get no warning about CG
        with gpytorch.settings.max_cg_iterations(200), warnings.catch_warnings(record=True) as w:
            res = lazy_tensor.inv_matmul(test_vector, test_left)
            actual = test_left_copy @ evaluated.inverse() @ test_vector_copy
            self.assertEqual(len(w), 0)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        # Make sure that we get no warning about CG
        with warnings.catch_warnings(record=True) as w:
            res.backward(gradient=grad)
            actual.backward(gradient=grad)
            self.assertEqual(len(w), 0)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )
        self.assertLess(
            ((test_left.grad - test_left_copy.grad).abs() / test_left.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )
        self.assertLess(
            ((test_vector.grad - test_vector_copy.grad).abs() / test_vector.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )

    def test_inv_matmul_matrix(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = lazy_tensor.eager_rhss[0].clone().detach().requires_grad_(True)
        test_vector_copy = lazy_tensor_copy.eager_rhss[0].clone().detach().requires_grad_(True)
        # Make sure that we get no warning about CG
        with gpytorch.settings.max_cg_iterations(100), warnings.catch_warnings(record=True) as w:
            res = lazy_tensor.inv_matmul(test_vector)
            actual = evaluated.inverse().matmul(test_vector_copy)
            self.assertEqual(len(w), 0)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        # Make sure that we get a warning that CG was run
        with warnings.catch_warnings(record=True) as w:
            res.backward(gradient=grad)
            actual.backward(gradient=grad)
            self.assertEqual(len(w), 1)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )
        self.assertLess(
            ((test_vector.grad - test_vector_copy.grad).abs() / test_vector.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )

    def test_inv_matmul_matrix_with_left(self):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = lazy_tensor.eager_rhss[0][..., 2:].clone().detach().requires_grad_(True)
        test_vector_copy = lazy_tensor_copy.eager_rhss[0][..., 2:].clone().detach().requires_grad_(True)
        test_left = lazy_tensor.eager_rhss[0][..., :2].transpose(-1, -2).clone().detach().requires_grad_(True)
        test_left_copy = lazy_tensor_copy.eager_rhss[0][..., :2].transpose(-1, -2).clone().detach().requires_grad_(True)
        # Make sure that we get no warning about CG
        with gpytorch.settings.max_cg_iterations(100), warnings.catch_warnings(record=True) as w:
            res = lazy_tensor.inv_matmul(test_vector, test_left)
            actual = test_left_copy @ evaluated.inverse() @ test_vector_copy
            self.assertEqual(len(w), 0)
        self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

        grad = torch.randn_like(res)
        # Make sure that we get no warning about CG
        with warnings.catch_warnings(record=True) as w:
            res.backward(gradient=grad)
            actual.backward(gradient=grad)
            self.assertEqual(len(w), 0)
        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )
        self.assertLess(
            ((test_left.grad - test_left_copy.grad).abs() / test_left.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )
        self.assertLess(
            ((test_vector.grad - test_vector_copy.grad).abs() / test_vector.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
        )

    def test_inv_quad_log_det(self):
        # Forward
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        flattened_evaluated = evaluated.view(-1, *lazy_tensor.matrix_shape)

        vecs = lazy_tensor.eager_rhss[0].clone().detach().requires_grad_(True)
        vecs_copy = lazy_tensor.eager_rhss[0].clone().detach().requires_grad_(True)

        with gpytorch.settings.num_trace_samples(128), warnings.catch_warnings(record=True) as w:
            res_inv_quad, res_log_det = lazy_tensor.inv_quad_log_det(inv_quad_rhs=vecs, log_det=True)
            self.assertEqual(len(w), 0)
        res = res_inv_quad + res_log_det

        actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2).sum(-1)
        actual_log_det = torch.cat(
            [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(lazy_tensor.batch_shape.numel())]
        ).view(lazy_tensor.batch_shape)
        actual = actual_inv_quad + actual_log_det

        diff = (res - actual).abs() / actual.abs().clamp(1, math.inf)
        self.assertLess(diff.max().item(), 15e-2)

    def test_inv_quad_log_det_no_reduce(self):
        # Forward
        lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        flattened_evaluated = evaluated.view(-1, *lazy_tensor.matrix_shape)

        vecs = lazy_tensor.eager_rhss[0].clone().detach().requires_grad_(True)
        vecs_copy = lazy_tensor.eager_rhss[0].clone().detach().requires_grad_(True)

        with gpytorch.settings.num_trace_samples(128), warnings.catch_warnings(record=True) as w:
            res_inv_quad, res_log_det = lazy_tensor.inv_quad_log_det(
                inv_quad_rhs=vecs, log_det=True, reduce_inv_quad=False
            )
            self.assertEqual(len(w), 0)
        res = res_inv_quad.sum(-1) + res_log_det

        actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2).sum(-1)
        actual_log_det = torch.cat(
            [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(lazy_tensor.batch_shape.numel())]
        ).view(lazy_tensor.batch_shape)
        actual = actual_inv_quad + actual_log_det

        diff = (res - actual).abs() / actual.abs().clamp(1, math.inf)
        self.assertLess(diff.max().item(), 15e-2)

    def test_root_inv_decomposition(self):
        lazy_tensor = self.create_lazy_tensor()
        root_approx = lazy_tensor.root_inv_decomposition()

        test_mat = lazy_tensor.eager_rhss[0].clone().detach()

        res = root_approx.matmul(test_mat)
        actual = lazy_tensor.inv_matmul(test_mat)
        self.assertLess(torch.norm(res - actual) / actual.norm(), 0.1)


class TestCachedCGLazyTensorBatch(TestCachedCGLazyTensor):
    seed = 0

    def create_lazy_tensor(self):
        mat = torch.randn(3, 5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)

        with gpytorch.settings.num_trace_samples(1000):  # For inv_quad_log_det tests
            lazy_tensor = NonLazyTensor(mat)
            eager_rhs = torch.randn(3, 5, 10).detach()
            solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLazyTensor.precompute_terms(
                lazy_tensor, eager_rhs.detach()
            )

        return CachedCGLazyTensor(
            lazy_tensor, [eager_rhs], [solve], probe_vecs, probe_vec_norms, probe_vec_solves, tmats
        )

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.base_lazy_tensor.tensor

    def test_inv_matmul_vec(self):
        pass

    def test_inv_matmul_vector_with_left(self):
        pass
