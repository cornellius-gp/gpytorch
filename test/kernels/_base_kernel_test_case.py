#!/usr/bin/env python3

from abc import abstractmethod
import torch


class BaseKernelTestCase(object):
    @abstractmethod
    def create_kernel_no_ard(self, **kwargs):
        raise NotImplementedError()

    def test_active_dims_list(self):
        kernel = self.create_kernel_no_ard(active_dims=[0, 2, 4, 6])
        x = torch.randn(50, 10)
        covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_basic = self.create_kernel_no_ard()
        covar_mat_actual = kernel_basic(x[:, [0, 2, 4, 6]]).evaluate_kernel().evaluate()

        self.assertLess(torch.norm(covar_mat - covar_mat_actual), 1e-5)

    def test_active_dims_range(self):
        active_dims = list(range(3, 9))
        kernel = self.create_kernel_no_ard(active_dims=active_dims)
        x = torch.randn(50, 10)
        covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_basic = self.create_kernel_no_ard()
        covar_mat_actual = kernel_basic(x[:, active_dims]).evaluate_kernel().evaluate()

        self.assertLess(torch.norm(covar_mat - covar_mat_actual), 1e-5)
