#!/usr/bin/env python3

import math
import unittest
from unittest.mock import MagicMock, patch

import linear_operator
import torch
from linear_operator import to_dense
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase

import gpytorch


class TestLazyEvaluatedKernelTensorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_op(self):
        kern = gpytorch.kernels.RBFKernel()
        mat1 = torch.randn(2, 5, 6)
        mat2 = mat1.detach().clone()
        return kern(mat1, mat2)

    def evaluate_linear_op(self, lazy_tensor):
        with gpytorch.settings.lazily_evaluate_kernels(False):
            return to_dense(lazy_tensor.kernel(lazy_tensor.x1, lazy_tensor.x2))

    def _test_matmul(self, rhs):
        lazy_tensor = self.create_linear_op().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_linear_op(lazy_tensor_copy)
        rhs_evaluated = to_dense(rhs)

        res = lazy_tensor.matmul(rhs)
        actual = evaluated.matmul(rhs_evaluated)
        res_evaluated = to_dense(res)
        self.assertAllClose(res_evaluated, actual)

        grad = torch.randn_like(res_evaluated)
        res_evaluated.backward(gradient=grad)
        actual.backward(gradient=grad)
        for param, param_copy in zip(lazy_tensor.kernel.parameters(), lazy_tensor_copy.kernel.parameters()):
            self.assertAllClose(param.grad, param_copy.grad, rtol=1e-3)
        self.assertAllClose(
            lazy_tensor.x1.grad + lazy_tensor.x2.grad, lazy_tensor_copy.x1.grad + lazy_tensor_copy.x2.grad, rtol=1e-3
        )

    def _test_rmatmul(self, lhs):
        lazy_tensor = self.create_linear_op().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_linear_op(lazy_tensor_copy)

        res = lhs @ lazy_tensor
        actual = lhs @ evaluated
        self.assertAllClose(res, actual)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for param, param_copy in zip(lazy_tensor.kernel.parameters(), lazy_tensor_copy.kernel.parameters()):
            self.assertAllClose(param.grad, param_copy.grad, rtol=1e-3)
        self.assertAllClose(
            lazy_tensor.x1.grad + lazy_tensor.x2.grad,
            lazy_tensor_copy.x1.grad + lazy_tensor_copy.x2.grad,
            rtol=1e-3,
            atol=1e-4,
        )

    def _test_inv_matmul(self, rhs, lhs=None, cholesky=False):
        lazy_tensor = self.create_linear_op().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_linear_op(lazy_tensor_copy)
        evaluated.register_hook(self._ensure_symmetric_grad)

        # Create a test right hand side and left hand side
        rhs.requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)
        if lhs is not None:
            lhs.requires_grad_(True)
            lhs_copy = lhs.clone().detach().requires_grad_(True)

        _wrapped_cg = MagicMock(wraps=linear_operator.utils.linear_cg)
        with patch("linear_operator.utils.linear_cg", new=_wrapped_cg) as linear_cg_mock:
            with gpytorch.settings.max_cholesky_size(math.inf if cholesky else 0), gpytorch.settings.cg_tolerance(1e-4):
                # Perform the inv_matmul
                if lhs is not None:
                    res = lazy_tensor.solve(rhs, lhs)
                    actual = lhs_copy @ evaluated.inverse() @ rhs_copy
                else:
                    res = lazy_tensor.solve(rhs)
                    actual = evaluated.inverse().matmul(rhs_copy)
                self.assertAllClose(res, actual, rtol=0.02, atol=1e-5)

                # Perform backward pass
                grad = torch.randn_like(res)
                res.backward(gradient=grad)
                actual.backward(gradient=grad)
                for param, param_copy in zip(lazy_tensor.kernel.parameters(), lazy_tensor_copy.kernel.parameters()):
                    self.assertAllClose(param.grad, param_copy.grad, rtol=1e-3)
                self.assertAllClose(
                    lazy_tensor.x1.grad + lazy_tensor.x2.grad,
                    lazy_tensor_copy.x1.grad + lazy_tensor_copy.x2.grad,
                    rtol=1e-3,
                )
                self.assertAllClose(rhs.grad, rhs_copy.grad, rtol=0.03, atol=1e-5)
                if lhs is not None:
                    self.assertAllClose(lhs.grad, lhs_copy.grad, rtol=0.03, atol=1e-5)

            # Determine if we've called CG or not
            if not cholesky and self.__class__.should_call_cg:
                self.assertTrue(linear_cg_mock.called)
            else:
                self.assertFalse(linear_cg_mock.called)

    def test_batch_getitem(self):
        """Indexing was wrong when the kernel had more batch dimensions than the
        data"""
        x1 = torch.randn(5, 6)
        x2 = torch.randn(5, 6)
        kern = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2]))
        k = kern(x1, x2)
        self.assertEqual(k.size(), torch.Size([2, 5, 5]))
        self.assertEqual(k[..., :4, :3].size(), torch.Size([2, 4, 3]))

    def test_batch_getitem_multioutput(self):
        """Ensure slicing is efficient when using a multioutput kernel"""
        x1 = torch.randn(5, 6)
        x2 = torch.randn(5, 6)
        kern = gpytorch.kernels.RBFKernelGrad(batch_shape=torch.Size([2]))
        k = kern(x1, x2)
        k.evaluate_kernel = MagicMock(name="evaluate_kernel")
        k_sliced = k[..., :7, :14]
        self.assertFalse(k.evaluate_kernel.called)
        self.assertEqual(k.size(), torch.Size([2, 35, 35]))
        self.assertEqual(k_sliced.size(), torch.Size([2, 7, 14]))

    def test_getitem_tensor_index(self):
        # Not supported a.t.m. with LazyEvaluatedKernelTensors
        pass

    def test_bilinear_derivative(self):
        pass

    def test_t_matmul_matrix(self):
        pass

    def test_half(self):
        # many transform operations aren't supported in half so we overwrite
        # this test
        lazy_tensor = self.create_linear_op()
        lazy_tensor.kernel.raw_lengthscale_constraint.transform = lambda x: x + 0.1
        self._test_half(lazy_tensor)

    def test_grad_state(self):
        k = gpytorch.kernels.RBFKernel()
        X = torch.randn(2, 3)
        X.requires_grad = True
        lazy_tensor = k(X)
        self.assertTrue(lazy_tensor.to_dense().requires_grad)
        with torch.no_grad():
            lazy_tensor = k(X)
        self.assertFalse(lazy_tensor.to_dense().requires_grad)


class TestLazyEvaluatedKernelTensorMultitaskBatch(TestLazyEvaluatedKernelTensorBatch):
    seed = 0
    skip_slq_tests = True  # we skip these because of the kronecker structure

    def create_linear_op(self):
        kern = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks=3, rank=2)
        mat1 = torch.randn(2, 5, 6)
        mat2 = mat1.detach().clone()
        return kern(mat1, mat2)

    def test_inv_matmul_matrix_with_checkpointing(self):
        pass

    def test_half(self):
        # many transform operations aren't supported in half so we overwrite
        # this test
        lazy_tensor = self.create_linear_op()
        lazy_tensor.kernel.data_covar_module.raw_lengthscale_constraint.transform = lambda x: x + 0.1
        self._test_half(lazy_tensor)


class TestLazyEvaluatedKernelTensorAdditive(TestLazyEvaluatedKernelTensorBatch):
    seed = 0

    def create_linear_op(self):
        kern = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.RBFKernel(), num_dims=6)
        mat1 = torch.randn(5, 6)
        mat2 = mat1.detach().clone()
        return kern(mat1, mat2)

    def evaluate_linear_op(self, lazy_tensor):
        res = to_dense(
            gpytorch.Module.__call__(
                lazy_tensor.kernel.base_kernel,
                lazy_tensor.x1.transpose(-1, -2).unsqueeze(-1),
                lazy_tensor.x2.transpose(-1, -2).unsqueeze(-1),
            )
        ).sum(0)
        return res

    def test_inv_matmul_matrix_with_checkpointing(self):
        pass

    def test_half(self):
        # many transform operations aren't supported in half so we overwrite
        # this test
        lazy_tensor = self.create_linear_op()
        lazy_tensor.kernel.base_kernel.raw_lengthscale_constraint.transform = lambda x: x + 0.1
        self._test_half(lazy_tensor)
