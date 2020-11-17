#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import SpectralMixtureKernel


class TestSpectralMixtureKernel(unittest.TestCase):
    def create_kernel(self, num_dims, **kwargs):
        return SpectralMixtureKernel(num_mixtures=5, ard_num_dims=num_dims, **kwargs)

    def create_data_no_batch(self):
        return torch.randn(50, 10)

    def create_data_single_batch(self):
        return torch.randn(2, 50, 2)

    def create_data_double_batch(self):
        return torch.randn(3, 2, 50, 2)

    def test_active_dims_list(self):
        x = self.create_data_no_batch()
        kernel = self.create_kernel(num_dims=4, active_dims=[0, 2, 4, 6])
        y = torch.randn_like(x[..., 0])
        kernel.initialize_from_data(x, y)
        kernel.initialize_from_data_empspect(x, y)

        covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_basic = self.create_kernel(num_dims=4)
        kernel_basic.raw_mixture_weights.data = kernel.raw_mixture_weights
        kernel_basic.raw_mixture_means.data = kernel.raw_mixture_means
        kernel_basic.raw_mixture_scales.data = kernel.raw_mixture_scales
        covar_mat_actual = kernel_basic(x[:, [0, 2, 4, 6]]).evaluate_kernel().evaluate()

        self.assertLess(torch.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(), 1e-4)

    def test_active_dims_range(self):
        active_dims = list(range(3, 9))
        x = self.create_data_no_batch()
        kernel = self.create_kernel(num_dims=6, active_dims=active_dims)
        y = torch.randn_like(x[..., 0])
        kernel.initialize_from_data(x, y)
        kernel.initialize_from_data_empspect(x, y)

        covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_basic = self.create_kernel(num_dims=6)
        kernel_basic.raw_mixture_weights.data = kernel.raw_mixture_weights
        kernel_basic.raw_mixture_means.data = kernel.raw_mixture_means
        kernel_basic.raw_mixture_scales.data = kernel.raw_mixture_scales
        covar_mat_actual = kernel_basic(x[:, active_dims]).evaluate_kernel().evaluate()

        self.assertLess(torch.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(), 1e-4)

    def test_no_batch_kernel_single_batch_x(self):
        x = self.create_data_single_batch()
        kernel = self.create_kernel(num_dims=x.size(-1))
        y = torch.randn_like(x[..., 0])
        kernel.initialize_from_data(x, y)
        kernel.initialize_from_data_empspect(x, y)
        batch_covar_mat = kernel(x).evaluate_kernel().evaluate()

        actual_mat_1 = kernel(x[0]).evaluate_kernel().evaluate()
        actual_mat_2 = kernel(x[1]).evaluate_kernel().evaluate()
        actual_covar_mat = torch.cat([actual_mat_1.unsqueeze(0), actual_mat_2.unsqueeze(0)])

        self.assertLess(torch.norm(batch_covar_mat - actual_covar_mat) / actual_covar_mat.norm(), 1e-4)

        # Test diagonal
        kernel_diag = kernel(x, diag=True)
        actual_diag = actual_covar_mat.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(kernel_diag - actual_diag) / actual_diag.norm(), 1e-4)

    def test_single_batch_kernel_single_batch_x(self):
        x = self.create_data_single_batch()
        kernel = self.create_kernel(num_dims=x.size(-1), batch_shape=torch.Size([]))
        y = torch.randn_like(x[..., 0])
        kernel.initialize_from_data(x, y)
        kernel.initialize_from_data_empspect(x, y)
        batch_covar_mat = kernel(x).evaluate_kernel().evaluate()

        actual_mat_1 = kernel(x[0]).evaluate_kernel().evaluate()
        actual_mat_2 = kernel(x[1]).evaluate_kernel().evaluate()
        actual_covar_mat = torch.cat([actual_mat_1.unsqueeze(0), actual_mat_2.unsqueeze(0)])

        self.assertLess(torch.norm(batch_covar_mat - actual_covar_mat) / actual_covar_mat.norm(), 1e-4)

        # Test diagonal
        kernel_diag = kernel(x, diag=True)
        actual_diag = actual_covar_mat.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(kernel_diag - actual_diag) / actual_diag.norm(), 1e-4)

    def test_smoke_double_batch_kernel_double_batch_x(self):
        x = self.create_data_double_batch()
        kernel = self.create_kernel(num_dims=x.size(-1), batch_shape=torch.Size([3, 2]))
        y = torch.randn_like(x[..., 0])
        kernel.initialize_from_data(x, y)
        kernel.initialize_from_data_empspect(x, y)

        batch_covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_diag = kernel(x, diag=True)
        return batch_covar_mat, kernel_diag

    def test_kernel_getitem_single_batch(self):
        x = self.create_data_single_batch()
        kernel = self.create_kernel(num_dims=x.size(-1), batch_shape=torch.Size([2]))

        res1 = kernel(x).evaluate()[0]  # Result of first kernel on first batch of data

        new_kernel = kernel[0]
        res2 = new_kernel(x[0]).evaluate()  # Should also be result of first kernel on first batch of data.

        self.assertLess(torch.norm(res1 - res2) / res1.norm(), 1e-4)

        # Test diagonal
        kernel_diag = kernel(x, diag=True)
        actual_diag = res1.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(kernel_diag - actual_diag) / actual_diag.norm(), 1e-4)

    def test_kernel_getitem_double_batch(self):
        x = self.create_data_double_batch()
        kernel = self.create_kernel(num_dims=x.size(-1), batch_shape=torch.Size([3, 2]))

        res1 = kernel(x).evaluate()[0, 1]  # Result of first kernel on first batch of data

        new_kernel = kernel[0, 1]
        res2 = new_kernel(x[0, 1]).evaluate()  # Should also be result of first kernel on first batch of data.

        self.assertLess(torch.norm(res1 - res2) / res1.norm(), 1e-4)

        # Test diagonal
        kernel_diag = kernel(x, diag=True)
        actual_diag = res1.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(kernel_diag - actual_diag) / actual_diag.norm(), 1e-4)


if __name__ == "__main__":
    unittest.main()
