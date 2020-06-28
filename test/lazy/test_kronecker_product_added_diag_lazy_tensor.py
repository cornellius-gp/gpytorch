#!/usr/bin/env python3

import unittest

import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel
from gpytorch.lazy import DiagLazyTensor, KroneckerProductAddedDiagLazyTensor, KroneckerProductLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase

# import numpy as np
# import gpytorch


# TODO: write unit test for the three classes - lazylogdet is here


class TestKroneckerProductAddedDiagLazyTensor(unittest.TestCase, LazyTensorTestCase):
    def create_lazy_tensor(self):
        dims = (3, 5, 10)
        latent_spaces = [torch.arange(d).float() for d in dims]
        kernels = [MaternKernel()(ll) for ll in latent_spaces]
        lazy_kernel_matrices = KroneckerProductLazyTensor(*kernels)

        return KroneckerProductAddedDiagLazyTensor(
            lazy_kernel_matrices, DiagLazyTensor(0.1 * torch.ones(lazy_kernel_matrices.shape[-1]))
        )

    def evaluate_lazy_tensor(self, lazy_tensor):
        tensor = lazy_tensor._lazy_tensor.evaluate()
        diag = lazy_tensor._diag_tensor._diag
        return tensor + diag.diag()

    def test_log_likelihood_computation(self):
        covariance = self.create_lazy_tensor()
        cloned_covariance = covariance.clone()

        mean_value = torch.zeros(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)
        dist = MultivariateNormal(mean_value, covariance)

        target = torch.randn_like(mean_value)
        value = dist.log_prob(target)
        value.backward()

        lengthscale_grads = torch.tensor(
            [lt.kernel.raw_lengthscale.grad.data for lt in covariance._lazy_tensor.lazy_tensors]
        )

        exact_dist = MultivariateNormal(mean_value, cloned_covariance.evaluate())
        exact_value = exact_dist.log_prob(target)
        exact_value.backward()
        exact_lengthscale_grads = torch.tensor(
            [lt.kernel.raw_lengthscale.grad.data for lt in cloned_covariance._lazy_tensor.lazy_tensors]
        )

        ll_relative_error = (value.data - exact_value.data).norm() / exact_value.data.norm()

        ll_grad_relative_error = (lengthscale_grads - exact_lengthscale_grads).norm() / exact_lengthscale_grads.norm()

        print(ll_grad_relative_error, lengthscale_grads, exact_lengthscale_grads)
        self.assertLessEqual(ll_relative_error, 1e-3)
        self.assertLessEqual(ll_grad_relative_error, 1e-2)


if __name__ == "__main__":
    unittest.main()
