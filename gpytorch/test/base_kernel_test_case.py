#!/usr/bin/env python3

import pickle
from abc import abstractmethod

import torch

from .base_test_case import BaseTestCase


class BaseKernelTestCase(BaseTestCase):
    @abstractmethod
    def create_kernel_no_ard(self, **kwargs):
        raise NotImplementedError()

    def create_kernel_ard(self, num_dims, **kwargs):
        raise NotImplementedError()

    def create_data_no_batch(self):
        return torch.randn(50, 10)

    def create_data_single_batch(self):
        return torch.randn(2, 3, 2)

    def create_data_double_batch(self):
        return torch.randn(3, 2, 50, 2)

    def test_active_dims_list(self):
        kernel = self.create_kernel_no_ard(active_dims=[0, 2, 4, 6])
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_basic = self.create_kernel_no_ard()
        covar_mat_actual = kernel_basic(x[:, [0, 2, 4, 6]]).evaluate_kernel().evaluate()

        self.assertAllClose(covar_mat, covar_mat_actual, rtol=1e-3, atol=1e-5)

    def test_active_dims_range(self):
        active_dims = list(range(3, 9))
        kernel = self.create_kernel_no_ard(active_dims=active_dims)
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel_basic = self.create_kernel_no_ard()
        covar_mat_actual = kernel_basic(x[:, active_dims]).evaluate_kernel().evaluate()

        self.assertAllClose(covar_mat, covar_mat_actual, rtol=1e-3, atol=1e-5)

    def test_no_batch_kernel_single_batch_x_no_ard(self):
        kernel = self.create_kernel_no_ard()
        x = self.create_data_single_batch()
        batch_covar_mat = kernel(x).evaluate_kernel().evaluate()

        actual_mat_1 = kernel(x[0]).evaluate_kernel().evaluate()
        actual_mat_2 = kernel(x[1]).evaluate_kernel().evaluate()
        actual_covar_mat = torch.cat([actual_mat_1.unsqueeze(0), actual_mat_2.unsqueeze(0)])

        self.assertAllClose(batch_covar_mat, actual_covar_mat, rtol=1e-3, atol=1e-5)

        # Test diagonal
        kernel_diag = kernel(x, diag=True)
        actual_diag = actual_covar_mat.diagonal(dim1=-1, dim2=-2)
        self.assertAllClose(kernel_diag, actual_diag, rtol=1e-3, atol=1e-5)

    def test_single_batch_kernel_single_batch_x_no_ard(self):
        kernel = self.create_kernel_no_ard(batch_shape=torch.Size([]))
        x = self.create_data_single_batch()
        batch_covar_mat = kernel(x).evaluate_kernel().evaluate()

        actual_mat_1 = kernel(x[0]).evaluate_kernel().evaluate()
        actual_mat_2 = kernel(x[1]).evaluate_kernel().evaluate()
        actual_covar_mat = torch.cat([actual_mat_1.unsqueeze(0), actual_mat_2.unsqueeze(0)])

        self.assertAllClose(batch_covar_mat, actual_covar_mat, rtol=1e-3, atol=1e-5)

        # Test diagonal
        kernel_diag = kernel(x, diag=True)
        actual_diag = actual_covar_mat.diagonal(dim1=-1, dim2=-2)
        self.assertAllClose(kernel_diag, actual_diag, rtol=1e-3, atol=1e-5)

    def test_no_batch_kernel_double_batch_x_no_ard(self):
        kernel = self.create_kernel_no_ard(batch_shape=torch.Size([]))
        x = self.create_data_double_batch()
        batch_covar_mat = kernel(x).evaluate_kernel().evaluate()

        ij_actual_covars = []
        for i in range(x.size(0)):
            i_actual_covars = []
            for j in range(x.size(1)):
                i_actual_covars.append(kernel(x[i, j]).evaluate_kernel().evaluate())
            ij_actual_covars.append(torch.cat([ac.unsqueeze(0) for ac in i_actual_covars]))

        actual_covar_mat = torch.cat([ac.unsqueeze(0) for ac in ij_actual_covars])

        self.assertAllClose(batch_covar_mat, actual_covar_mat, rtol=1e-3, atol=1e-5)

        # Test diagonal
        kernel_diag = kernel(x, diag=True)
        actual_diag = actual_covar_mat.diagonal(dim1=-1, dim2=-2)
        self.assertAllClose(kernel_diag, actual_diag, rtol=1e-3, atol=1e-5)

    def test_no_batch_kernel_double_batch_x_ard(self):
        try:
            kernel = self.create_kernel_ard(num_dims=2, batch_shape=torch.Size([]))
        except NotImplementedError:
            return

        x = self.create_data_double_batch()
        batch_covar_mat = kernel(x).evaluate_kernel().evaluate()

        ij_actual_covars = []
        for i in range(x.size(0)):
            i_actual_covars = []
            for j in range(x.size(1)):
                i_actual_covars.append(kernel(x[i, j]).evaluate_kernel().evaluate())
            ij_actual_covars.append(torch.cat([ac.unsqueeze(0) for ac in i_actual_covars]))

        actual_covar_mat = torch.cat([ac.unsqueeze(0) for ac in ij_actual_covars])

        self.assertAllClose(batch_covar_mat, actual_covar_mat, rtol=1e-3, atol=1e-5)

        # Test diagonal
        kernel_diag = kernel(x, diag=True)
        actual_diag = actual_covar_mat.diagonal(dim1=-1, dim2=-2)
        self.assertAllClose(kernel_diag, actual_diag, rtol=1e-3, atol=1e-5)

    def test_smoke_double_batch_kernel_double_batch_x_no_ard(self):
        kernel = self.create_kernel_no_ard(batch_shape=torch.Size([3, 2]))
        x = self.create_data_double_batch()
        batch_covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel(x, diag=True)
        return batch_covar_mat

    def test_smoke_double_batch_kernel_double_batch_x_ard(self):
        try:
            kernel = self.create_kernel_ard(num_dims=2, batch_shape=torch.Size([3, 2]))
        except NotImplementedError:
            return

        x = self.create_data_double_batch()
        batch_covar_mat = kernel(x).evaluate_kernel().evaluate()
        kernel(x, diag=True)
        return batch_covar_mat

    def test_kernel_getitem_single_batch(self):
        kernel = self.create_kernel_no_ard(batch_shape=torch.Size([2]))
        x = self.create_data_single_batch()

        res1 = kernel(x).evaluate()[0]  # Result of first kernel on first batch of data

        new_kernel = kernel[0]
        res2 = new_kernel(x[0]).evaluate()  # Should also be result of first kernel on first batch of data.

        self.assertAllClose(res1, res2, rtol=1e-3, atol=1e-5)

    def test_kernel_getitem_double_batch(self):
        kernel = self.create_kernel_no_ard(batch_shape=torch.Size([3, 2]))
        x = self.create_data_double_batch()

        res1 = kernel(x).evaluate()[0, 1]  # Result of first kernel on first batch of data

        new_kernel = kernel[0, 1]
        res2 = new_kernel(x[0, 1]).evaluate()  # Should also be result of first kernel on first batch of data.

        self.assertAllClose(res1, res2, rtol=1e-3, atol=1e-5)

    def test_kernel_pickle_unpickle(self):
        kernel = self.create_kernel_no_ard(batch_shape=torch.Size([]))
        pickle.loads(pickle.dumps(kernel))  # Should be able to pickle and unpickle a kernel
