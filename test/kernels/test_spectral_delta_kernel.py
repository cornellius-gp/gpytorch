import unittest

import torch

from gpytorch.kernels import SpectralDeltaKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestSpectralDeltaKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, num_dims=2, **kwargs):
        return SpectralDeltaKernel(num_dims=num_dims, num_deltas=64, **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return SpectralDeltaKernel(num_dims=num_dims, ard_num_dims=num_dims, **kwargs)

    def create_data_no_batch(self, num_dims=2):
        return torch.randn(50, num_dims)

    def create_data_single_batch(self):
        return torch.randn(2, 50, 2)

    def create_data_double_batch(self):
        return torch.randn(3, 2, 50, 2)

    def test_active_dims_list(self):
        """
        Overwrite because this kernel needs to know how many dims it will be operating on.
        """
        kernel = self.create_kernel_no_ard(num_dims=4, active_dims=[0, 2, 4, 6])
        x = self.create_data_no_batch(num_dims=10)
        covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_basic = self.create_kernel_no_ard(num_dims=4)
        kernel_basic.Z = kernel.Z
        covar_mat_actual = kernel_basic(x[:, [0, 2, 4, 6]]).evaluate_kernel().evaluate()

        self.assertLess(torch.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(), 1e-4)

    def test_active_dims_range(self):
        """
        Overwrite because this kernel needs to know how many dims it will be operating on.
        """
        active_dims = list(range(3, 9))
        kernel = self.create_kernel_no_ard(num_dims=len(active_dims), active_dims=active_dims)
        x = self.create_data_no_batch(num_dims=10)
        covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_basic = self.create_kernel_no_ard(num_dims=len(active_dims))
        kernel_basic.Z = kernel.Z
        covar_mat_actual = kernel_basic(x[:, active_dims]).evaluate_kernel().evaluate()

        self.assertLess(torch.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(), 1e-4)
