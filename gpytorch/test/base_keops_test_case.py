from abc import abstractmethod
from unittest.mock import patch

import torch

import gpytorch
from .base_test_case import BaseTestCase


CHOLESKY_SIZE_KEOPS, CHOLESKY_SIZE_NONKEOPS = 2, 800


class BaseKeOpsTestCase(BaseTestCase):
    @abstractmethod
    def k1(self):
        """Returns first kernel class"""
        pass

    @abstractmethod
    def k2(self):
        """Returns second kernel class"""
        pass

    # tests the keops implementation
    def test_forward_x1_eq_x2(self, ard=False, use_keops=True, **kwargs):
        max_cholesky_size = CHOLESKY_SIZE_KEOPS if use_keops else CHOLESKY_SIZE_NONKEOPS
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            ndims = 3
            x1 = torch.randn(100, 3)

            if ard:
                kern1 = self.k1(ard_num_dims=ndims, **kwargs)
                kern2 = self.k2(ard_num_dims=ndims, **kwargs)
            else:
                kern1 = self.k1(**kwargs)
                kern2 = self.k2(**kwargs)

            # The patch makes sure that we're actually using KeOps
            # However, we're going to bypass KeOps and instead just use non-LazyTensors
            with patch("gpytorch.kernels.keops.keops_kernel.LazyTensor", wraps=lambda x: x) as keops_mock:
                k1 = kern1(x1, x1).to_dense()
                k2 = kern2(x1, x1).to_dense()
                self.assertLess(torch.norm(k1 - k2), 1e-4)

                # Test diagonal
                d1 = kern1(x1, x1).diagonal(dim1=-1, dim2=-2)
                d2 = kern2(x1, x1).diagonal(dim1=-1, dim2=-2)
                self.assertLess(torch.norm(d1 - d2), 1e-4)
                self.assertTrue(torch.equal(k1.diag(), d1))

        if use_keops:
            self.assertTrue(keops_mock.called)

    def test_forward_x1_eq_x2_ard(self):
        return self.test_forward_x1_eq_x2(ard=True)

    def test_forward_x1_neq_x2(self, use_keops=True, ard=False, **kwargs):
        max_cholesky_size = CHOLESKY_SIZE_KEOPS if use_keops else CHOLESKY_SIZE_NONKEOPS
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            ndims = 3
            x1 = torch.randn(100, ndims)
            x2 = torch.randn(50, ndims)

            if ard:
                kern1 = self.k1(ard_num_dims=ndims, **kwargs)
                kern2 = self.k2(ard_num_dims=ndims, **kwargs)
            else:
                kern1 = self.k1(**kwargs)
                kern2 = self.k2(**kwargs)

            with patch("gpytorch.kernels.keops.keops_kernel.LazyTensor", wraps=lambda x: x) as keops_mock:
                # The patch makes sure that we're actually using KeOps
                k1 = kern1(x1, x2).to_dense()
                k2 = kern2(x1, x2).to_dense()
                self.assertLess(torch.norm(k1 - k2), 1e-3)

                # Test diagonal
                d1 = kern1(x1, x1).diagonal(dim1=-1, dim2=-2)
                d2 = kern2(x1, x1).diagonal(dim1=-1, dim2=-2)
                self.assertLess(torch.norm(d1 - d2), 1e-4)

        if use_keops:
            self.assertTrue(keops_mock.called)

    def test_forward_x1_meq_x2_ard(self):
        return self.test_forward_x1_neq_x2(ard=True)

    def test_batch_matmul(self, use_keops=True, **kwargs):
        max_cholesky_size = CHOLESKY_SIZE_KEOPS if use_keops else CHOLESKY_SIZE_NONKEOPS
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            x1 = torch.randn(3, 2, 100, 3)
            kern1 = self.k1(**kwargs)
            kern2 = self.k2(**kwargs)

            rhs = torch.randn(3, 2, 100, 1)
            with patch("gpytorch.kernels.keops.keops_kernel.LazyTensor", wraps=lambda x: x) as keops_mock:
                # The patch makes sure that we're actually using KeOps
                res1 = kern1(x1, x1).matmul(rhs)
                res2 = kern2(x1, x1).matmul(rhs)
                self.assertLess(torch.norm(res1 - res2), 1e-3)

        if use_keops:
            self.assertTrue(keops_mock.called)

    def test_gradient(self, use_keops=True, ard=False, **kwargs):
        max_cholesky_size = CHOLESKY_SIZE_KEOPS if use_keops else CHOLESKY_SIZE_NONKEOPS
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            ndims = 3

            x1 = torch.randn(4, 100, ndims)

            if ard:
                kern1 = self.k1(ard_num_dims=ndims, **kwargs)
                kern2 = self.k2(ard_num_dims=ndims, **kwargs)
            else:
                kern1 = self.k1(**kwargs)
                kern2 = self.k2(**kwargs)

            with patch("gpytorch.kernels.keops.keops_kernel.LazyTensor", wraps=lambda x: x) as keops_mock:
                # The patch makes sure that we're actually using KeOps
                res1 = kern1(x1, x1)
                res2 = kern2(x1, x1)
                s1 = res1.sum()
                s2 = res2.sum()

                # stack all gradients into a tensor
                grad_s1 = torch.vstack(torch.autograd.grad(s1, [*kern1.hyperparameters()]))
                grad_s2 = torch.vstack(torch.autograd.grad(s2, [*kern2.hyperparameters()]))
                self.assertAllClose(grad_s1, grad_s2, rtol=1e-3, atol=1e-3)

        if use_keops:
            self.assertTrue(keops_mock.called)

    def test_gradient_ard(self):
        return self.test_gradient(ard=True)

    # tests the nonkeops implementation (_nonkeops_covar_func)
    def test_forward_x1_eq_x2_nonkeops(self):
        self.test_forward_x1_eq_x2(use_keops=False)

    def test_forward_x1_eq_x2_nonkeops_ard(self):
        self.test_forward_x1_eq_x2(use_keops=False, ard=True)

    def test_forward_x1_neq_x2_nonkeops(self):
        self.test_forward_x1_neq_x2(use_keops=False)

    def test_forward_x1_neq_x2_nonkeops_ard(self):
        self.test_forward_x1_neq_x2(use_keops=False, ard=True)

    def test_batch_matmul_nonkeops(self):
        self.test_batch_matmul(use_keops=False)

    def test_gradient_nonkeops(self):
        self.test_gradient(use_keops=False)

    def test_gradient_nonkeops_ard(self):
        self.test_gradient(use_keops=False, ard=True)
