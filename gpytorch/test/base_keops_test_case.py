from abc import abstractmethod

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

    def test_forward_x1_eq_x2(self, max_cholesky_size=2, ard=False, **kwargs):

        with gpytorch.settings.max_cholesky_size(max_cholesky_size):

            ndims = 3

            x1 = torch.randn(100, 3)

            if ard:
                kern1 = self.k1(ard_num_dims=ndims, **kwargs)
                kern2 = self.k2(ard_num_dims=ndims, **kwargs)
            else:
                kern1 = self.k1(**kwargs)
                kern2 = self.k2(**kwargs)

            k1 = kern1(x1, x1).to_dense()
            k2 = kern2(x1, x1).to_dense()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

    def test_forward_x1_eq_x2_ard(self):
        return self.test_forward_x1_eq_x2(ard=True)

    def test_forward_x1_neq_x2(self, max_cholesky_size=2, ard=False, **kwargs):

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

            k1 = kern1(x1, x2).to_dense()
            k2 = kern2(x1, x2).to_dense()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

    def test_forward_x1_meq_x2_ard(self):
        return self.test_forward_x1_neq_x2(ard=True)

    def test_batch_matmul(self, max_cholesky_size=2, **kwargs):

        with gpytorch.settings.max_cholesky_size(max_cholesky_size):

            x1 = torch.randn(3, 2, 100, 3)
            kern1 = self.k1(**kwargs)
            kern2 = self.k2(**kwargs)

            rhs = torch.randn(3, 2, 100, 1)
            res1 = kern1(x1, x1).matmul(rhs)
            res2 = kern2(x1, x1).matmul(rhs)

            self.assertLess(torch.norm(res1 - res2), 1e-4)

    def test_gradient(self, max_cholesky_size=2, ard=False, **kwargs):

        with gpytorch.settings.max_cholesky_size(max_cholesky_size):

            ndims = 3

            x1 = torch.randn(4, 100, ndims)

            if ard:
                kern1 = self.k1(ard_num_dims=ndims, **kwargs)
                kern2 = self.k2(ard_num_dims=ndims, **kwargs)
            else:
                kern1 = self.k1(**kwargs)
                kern2 = self.k2(**kwargs)

            res1 = kern1(x1, x1)
            res2 = kern2(x1, x1)

            s1 = res1.sum()
            s2 = res2.sum()

            # stack all gradients into a tensor
            grad_s1 = torch.vstack(torch.autograd.grad(s1, [*kern1.hyperparameters()]))
            grad_s2 = torch.vstack(torch.autograd.grad(s2, [*kern2.hyperparameters()]))

            self.assertAllClose(grad_s1, grad_s2, rtol=1e-4, atol=1e-5)

    def test_gradient_ard(self):
        return self.test_gradient(ard=True)

    # tests the nonkeops implementation (_nonkeops_covar_func)
    def test_forward_x1_eq_x2_nonkeops(self):
        self.test_forward_x1_eq_x2(max_cholesky_size=CHOLESKY_SIZE_NONKEOPS)

    def test_forward_x1_eq_x2_nonkeops_ard(self):
        self.test_forward_x1_eq_x2(max_cholesky_size=CHOLESKY_SIZE_NONKEOPS, ard=True)

    def test_forward_x1_neq_x2_nonkeops(self):
        self.test_forward_x1_neq_x2(max_cholesky_size=CHOLESKY_SIZE_NONKEOPS)

    def test_forward_x1_neq_x2_nonkeops_ard(self):
        self.test_forward_x1_neq_x2(max_cholesky_size=CHOLESKY_SIZE_NONKEOPS, ard=True)

    def test_batch_matmul_nonkeops(self):
        self.test_batch_matmul(max_cholesky_size=CHOLESKY_SIZE_NONKEOPS)

    def test_gradient_nonkeops(self):
        self.test_gradient(max_cholesky_size=CHOLESKY_SIZE_NONKEOPS)

    def test_gradient_nonkeops_ard(self):
        self.test_gradient(max_cholesky_size=CHOLESKY_SIZE_NONKEOPS, ard=True)
