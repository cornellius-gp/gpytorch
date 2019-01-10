#!/usr/bin/env python3

import unittest
import torch
from gpytorch.lazy import CholLazyTensor, SymmetricKernelInterpolatedLazyTensor, RootLazyTensor
from gpytorch.kernels import RBFKernel
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


class TestSymmetricKernelInterpolatedLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 1
    should_test_sample = True

    def create_lazy_tensor(self):
        kernel = RBFKernel()
        induc = torch.randn(6, 1)
        data = induc[..., 1:-1, :] + 0.01

        base_lazy_tensor = RootLazyTensor(torch.randn(6, 6, requires_grad=True))
        induc_induc_covar = CholLazyTensor(kernel(induc).evaluate().cholesky().detach().requires_grad_(True))
        data_induc_covar = kernel(data, induc).evaluate().detach().requires_grad_(True)
        return SymmetricKernelInterpolatedLazyTensor(
            base_lazy_tensor,
            induc_induc_covar.add_jitter(),
            data_induc_covar,
        )

    def evaluate_lazy_tensor(self, lazy_tensor):
        inv_product = lazy_tensor.induc_induc_covar.inv_matmul(lazy_tensor.induc_data_covar)
        return inv_product.transpose(-1, -2) @ (lazy_tensor.base_lazy_tensor @ inv_product)


class TestSymmetricKernelInterpolatedLazyTensorBatch(TestSymmetricKernelInterpolatedLazyTensor):
    seed = 1
    should_test_sample = True

    def create_lazy_tensor(self):
        kernel = RBFKernel()
        induc = torch.randn(2, 5, 1)
        data = induc[..., 1:-1, :] + 0.01
        half_mat = torch.randn(2, 5, 5)

        base_lazy_tensor = half_mat @ half_mat.transpose(-1, -2)
        induc_induc_covar = CholLazyTensor(kernel(induc).evaluate().cholesky().detach().requires_grad_(True))
        data_induc_covar = kernel(data, induc).evaluate().detach().requires_grad_(True)
        return SymmetricKernelInterpolatedLazyTensor(
            base_lazy_tensor,
            induc_induc_covar.add_jitter(),
            data_induc_covar,
        )

    def evaluate_lazy_tensor(self, lazy_tensor):
        inv_product = lazy_tensor.induc_induc_covar.inv_matmul(lazy_tensor.induc_data_covar)
        return inv_product.transpose(-1, -2) @ (lazy_tensor.base_lazy_tensor @ inv_product)


if __name__ == "__main__":
    unittest.main()
