#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import torch

import gpytorch
from gpytorch.lazy import RootLazyTensor
from gpytorch.test.base_test_case import BaseTestCase


class TestInvQuadLogDetNonBatch(BaseTestCase, unittest.TestCase):
    seed = 0
    matrix_shape = torch.Size((4, 4))

    def _test_inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, improper_logdet=False, add_diag=False):
        # Set up
        mat = torch.randn(*self.__class__.matrix_shape).requires_grad_(True)
        mat_clone = mat.detach().clone().requires_grad_(True)

        if inv_quad_rhs is not None:
            inv_quad_rhs.requires_grad_(True)
            inv_quad_rhs_clone = inv_quad_rhs.detach().clone().requires_grad_(True)

        # Compute actual values
        actual_tensor = mat_clone @ mat_clone.transpose(-1, -2)

        if add_diag:
            actual_tensor.diagonal(dim1=-2, dim2=-1).add_(1.0)

        if inv_quad_rhs is not None:
            actual_inv_quad = actual_tensor.inverse().matmul(inv_quad_rhs_clone).mul(inv_quad_rhs_clone)
            actual_inv_quad = actual_inv_quad.sum([-1, -2]) if inv_quad_rhs.dim() >= 2 else actual_inv_quad.sum()
        if logdet:
            flattened_tensor = actual_tensor.view(-1, *actual_tensor.shape[-2:])
            logdets = torch.cat([mat.logdet().unsqueeze(0) for mat in flattened_tensor])
            if actual_tensor.dim() > 2:
                actual_logdet = logdets.view(*actual_tensor.shape[:-2])
            else:
                actual_logdet = logdets.squeeze()

        # Compute values with LazyTensor
        _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
        with gpytorch.settings.num_trace_samples(2000), gpytorch.settings.max_cholesky_size(
            0
        ), gpytorch.settings.cg_tolerance(1e-5), gpytorch.settings.skip_logdet_forward(improper_logdet), patch(
            "gpytorch.utils.linear_cg", new=_wrapped_cg
        ) as linear_cg_mock:
            lazy_tensor = RootLazyTensor(mat)

            if add_diag:
                lazy_tensor = lazy_tensor.add_jitter(1.0)

            res_inv_quad, res_logdet = lazy_tensor.inv_quad_logdet(inv_quad_rhs=inv_quad_rhs, logdet=logdet)

        # Compare forward pass
        if inv_quad_rhs is not None:
            self.assertAllClose(res_inv_quad, actual_inv_quad, rtol=1e-2)
        if logdet:
            if improper_logdet:
                self.assertAlmostEqual(res_logdet.norm().item(), 0)
            else:
                self.assertAllClose(res_logdet, actual_logdet, rtol=1e-1, atol=2e-1)

        # Backward
        if inv_quad_rhs is not None:
            actual_inv_quad.sum().backward(retain_graph=True)
            res_inv_quad.sum().backward(retain_graph=True)
        if logdet:
            actual_logdet.sum().backward()
            res_logdet.sum().backward()

        self.assertAllClose(mat_clone.grad, mat.grad, rtol=1e-1, atol=2e-1)
        if inv_quad_rhs is not None:
            self.assertAllClose(inv_quad_rhs.grad, inv_quad_rhs_clone.grad, rtol=1e-2)

        # Make sure CG was called
        self.assertTrue(linear_cg_mock.called)

    def test_inv_quad_logdet_vector(self):
        rhs = torch.randn(self.matrix_shape[-1])
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True)

    def test_precond_inv_quad_logdet_vector(self):
        rhs = torch.randn(self.matrix_shape[-1])
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True, add_diag=True)

    def test_inv_quad_only_vector(self):
        rhs = torch.randn(self.matrix_shape[-1])
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=False)

    def test_precond_inv_quad_only_vector(self):
        rhs = torch.randn(self.matrix_shape[-1])
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=False, add_diag=True)

    def test_inv_quad_logdet_many_vectors(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True)

    def test_precond_inv_quad_logdet_many_vectors(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True, add_diag=True)

    def test_inv_quad_logdet_many_vectors_improper(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True, improper_logdet=True)

    def test_precond_inv_quad_logdet_many_vectors_improper(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=True, improper_logdet=True, add_diag=True)

    def test_inv_quad_only_many_vectors(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=False)

    def test_precond_inv_quad_only_many_vectors(self):
        rhs = torch.randn(*self.matrix_shape[:-1], 5)
        self._test_inv_quad_logdet(inv_quad_rhs=rhs, logdet=False, add_diag=True)


class TestInvQuadLogDetBatch(TestInvQuadLogDetNonBatch):
    seed = 0
    matrix_shape = torch.Size((3, 4, 4))

    def test_inv_quad_logdet_vector(self):
        pass

    def test_precond_inv_quad_logdet_vector(self):
        pass

    def test_inv_quad_only_vector(self):
        pass

    def test_precond_inv_quad_only_vector(self):
        pass


class TestInvQuadLogDetMultiBatch(TestInvQuadLogDetBatch):
    seed = 0
    matrix_shape = torch.Size((2, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()
