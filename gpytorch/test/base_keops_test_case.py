import gpytorch
import torch
from abc import ABC, abstractmethod

class BaseKeOpsTestCase(ABC):

    @abstractmethod
    def k1(self):
        """ Returns first kernel class"""
        pass

    @abstractmethod
    def k2(self):
        """ Returns second kernel class"""
        pass

        # tests the keops implementation
    def test_forward_x1_eq_x2(self, **kwargs):

        with gpytorch.settings.max_cholesky_size(2):

            x1 = torch.randn(100, 3)

            kern1 = self.k1(**kwargs)
            kern2 = self.k2(**kwargs)

            k1 = kern1(x1, x1).to_dense()
            k2 = kern2(x1, x1).to_dense()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

    def test_forward_x1_neq_x2(self, **kwargs):

        with gpytorch.settings.max_cholesky_size(2):

            x1 = torch.randn(100, 3)
            x2 = torch.randn(50, 3)

            kern1 = self.k1(**kwargs)
            kern2 = self.k2(**kwargs)

            k1 = kern1(x1, x2).to_dense()
            k2 = kern2(x1, x2).to_dense()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

    def test_batch_matmul(self, **kwargs):

        with gpytorch.settings.max_cholesky_size(2):

            x1 = torch.randn(3, 2, 100, 3)
            kern1 = self.k1(**kwargs)
            kern2 = self.k2(**kwargs)

            rhs = torch.randn(3, 2, 100, 1)
            res1 = kern1(x1, x1).matmul(rhs)
            res2 = kern2(x1, x1).matmul(rhs)

            self.assertLess(torch.norm(res1 - res2), 1e-4)

    # tests the nonkeops implementation (_nonkeops_covar_func)
    def test_forward_x1_eq_x2_nonkeops(self, **kwargs):

        with gpytorch.settings.max_cholesky_size(800):

            x1 = torch.randn(100, 3)

            kern1 = self.k1(**kwargs)
            kern2 = self.k2(**kwargs)

            k1 = kern1(x1, x1).to_dense()
            k2 = kern2(x1, x1).to_dense()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

    def test_forward_x1_neq_x2_nonkeops(self, **kwargs):

        with gpytorch.settings.max_cholesky_size(800):

            x1 = torch.randn(100, 3)
            x2 = torch.randn(50, 3)

            kern1 = self.k1(**kwargs)
            kern2 = self.k2(**kwargs)

            k1 = kern1(x1, x2).to_dense()
            k2 = kern2(x1, x2).to_dense()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

    def test_batch_matmul_nonkeops(self, **kwargs):

        with gpytorch.settings.max_cholesky_size(800):

            x1 = torch.randn(3, 2, 100, 3)
            kern1 = self.k1(**kwargs)
            kern2 = self.k2(**kwargs)

            rhs = torch.randn(3, 2, 100, 1)
            res1 = kern1(x1, x1).matmul(rhs)
            res2 = kern2(x1, x1).matmul(rhs)

            self.assertLess(torch.norm(res1 - res2), 1e-4)