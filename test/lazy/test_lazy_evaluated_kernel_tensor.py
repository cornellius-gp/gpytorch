#!/usr/bin/env python3

import torch
import unittest
import gpytorch
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestLazyEvaluatedKernelTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        kern = gpytorch.kernels.RBFKernel()
        mat = torch.randn(2, 5, 6)
        return kern(mat)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return gpytorch.lazy.delazify(gpytorch.Module.__call__(
            lazy_tensor.kernel,
            lazy_tensor.x1,
            lazy_tensor.x2
        ))

    def test_inv_matmul_matrix_with_checkpointing(self):
        # Add one checkpointing test
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        test_vector = torch.randn(2, 5, 3)
        test_vector_copy = test_vector.clone()
        with gpytorch.beta_features.checkpoint_kernel(2):
            res = lazy_tensor.inv_matmul(test_vector)
            actual = evaluated.inverse().matmul(test_vector_copy)
            self.assertLess(((res - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 3e-1)

            grad = torch.randn_like(res)
            res.backward(gradient=grad)
            actual.backward(gradient=grad)

        for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
            if arg_copy.grad is not None:
                self.assertLess(
                    ((arg.grad - arg_copy.grad).abs() / arg_copy.grad.abs().clamp(1, 1e5)).max().item(), 3e-1
                )

    def test_getitem_tensor_index(self):
        # Not supported a.t.m. with LazyEvaluatedKernelTensors
        pass

    def test_quad_form_derivative(self):
        pass


class TestLazyEvaluatedKernelTensorMultitaskBatch(TestLazyEvaluatedKernelTensorBatch):
    seed = 0

    def create_lazy_tensor(self):
        kern = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks=3, rank=2)
        mat = torch.randn(2, 5, 6)
        return kern(mat)

    def test_inv_matmul_matrix_with_checkpointing(self):
        pass


class TestLazyEvaluatedKernelTensorAdditive(TestLazyEvaluatedKernelTensorBatch):
    seed = 0

    def create_lazy_tensor(self):
        kern = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.RBFKernel(), num_dims=6)
        mat = torch.randn(5, 6)
        return kern(mat)

    def evaluate_lazy_tensor(self, lazy_tensor):
        res = gpytorch.lazy.delazify(gpytorch.Module.__call__(
            lazy_tensor.kernel.base_kernel,
            lazy_tensor.x1.transpose(-1, -2).unsqueeze(-1),
            lazy_tensor.x2.transpose(-1, -2).unsqueeze(-1)
        )).sum(0)
        return res

    def test_inv_matmul_matrix_with_checkpointing(self):
        pass
