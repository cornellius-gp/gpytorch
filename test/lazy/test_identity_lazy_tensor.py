#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import IdentityLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestIdentityLazyTensor(LazyTensorTestCase, unittest.TestCase):
    def _test_matmul(self, rhs):
        lazy_tensor = self.create_lazy_tensor().detach().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        res = lazy_tensor.matmul(rhs)
        actual = evaluated.matmul(rhs)
        self.assertAllClose(res, actual)

    def _test_rmatmul(self, lhs):
        lazy_tensor = self.create_lazy_tensor().detach().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        res = lhs @ lazy_tensor
        actual = lhs @ evaluated
        self.assertAllClose(res, actual)

    def _test_inv_matmul(self, rhs, lhs=None, cholesky=False):
        lazy_tensor = self.create_lazy_tensor().detach().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        # Create a test right hand side and left hand side
        rhs.requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)
        if lhs is not None:
            lhs.requires_grad_(True)
            lhs_copy = lhs.clone().detach().requires_grad_(True)
        if lhs is not None:
            res = lazy_tensor.inv_matmul(rhs, lhs)
            actual = lhs_copy @ evaluated.inverse() @ rhs_copy
        else:
            res = lazy_tensor.inv_matmul(rhs)
            actual = evaluated.inverse().matmul(rhs_copy)
        self.assertAllClose(res, actual, **self.tolerances["inv_matmul"])

    def _test_inv_quad_logdet(self, reduce_inv_quad=True, cholesky=False, lazy_tensor=None):
        if lazy_tensor is None:
            lazy_tensor = self.create_lazy_tensor()
        evaluated = self.evaluate_lazy_tensor(lazy_tensor)
        flattened_evaluated = evaluated.view(-1, *lazy_tensor.matrix_shape)

        vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 3, requires_grad=True)
        vecs_copy = vecs.clone().detach().requires_grad_(True)
        res_inv_quad, res_logdet = lazy_tensor.inv_quad_logdet(
            inv_quad_rhs=vecs, logdet=True, reduce_inv_quad=reduce_inv_quad
        )

        actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2)
        if reduce_inv_quad:
            actual_inv_quad = actual_inv_quad.sum(-1)
        actual_logdet = torch.cat(
            [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(lazy_tensor.batch_shape.numel())]
        ).view(lazy_tensor.batch_shape)

        self.assertAllClose(res_inv_quad, actual_inv_quad, **self.tolerances["inv_quad"])
        self.assertAllClose(res_logdet, actual_logdet, **self.tolerances["logdet"])

    def create_lazy_tensor(self):
        return IdentityLazyTensor(5)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return torch.eye(5)

    def test_diagonalization(self, symeig=False):
        lazy_tensor = self.create_lazy_tensor()
        evals, evecs = lazy_tensor.diagonalization()
        self.assertAllClose(evals, torch.ones(lazy_tensor.shape[:-1]))
        self.assertAllClose(evecs.evaluate(), torch.eye(lazy_tensor.size(-1)).expand(lazy_tensor.shape))

    def test_exp(self):
        lazy_tensor = self.create_lazy_tensor()
        exp = lazy_tensor.exp().evaluate()
        self.assertAllClose(exp, torch.eye(lazy_tensor.size(-1)).expand(*lazy_tensor.shape))

    def test_log(self):
        lazy_tensor = self.create_lazy_tensor()
        log = lazy_tensor.log().evaluate()
        self.assertAllClose(log, torch.zeros(*lazy_tensor.shape))

    def test_sqrt_inv_matmul(self):
        lazy_tensor = self.create_lazy_tensor().detach().requires_grad_(True)
        if len(lazy_tensor.batch_shape):
            return

        lazy_tensor_copy = lazy_tensor.clone().detach().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        # Create a test right hand side and left hand side
        rhs = torch.randn(*lazy_tensor.shape[:-1], 3).requires_grad_(True)
        lhs = torch.randn(*lazy_tensor.shape[:-2], 2, lazy_tensor.size(-1)).requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)
        lhs_copy = lhs.clone().detach().requires_grad_(True)

        # Perform forward pass
        sqrt_inv_matmul_res, inv_quad_res = lazy_tensor.sqrt_inv_matmul(rhs, lhs)
        evals, evecs = torch.linalg.eigh(evaluated)
        matrix_inv_root = evecs @ (evals.sqrt().reciprocal().unsqueeze(-1) * evecs.transpose(-1, -2))
        sqrt_inv_matmul_actual = lhs_copy @ matrix_inv_root @ rhs_copy
        inv_quad_actual = (lhs_copy @ matrix_inv_root).pow(2).sum(dim=-1)

        # Check forward pass
        self.assertAllClose(sqrt_inv_matmul_res, sqrt_inv_matmul_actual, **self.tolerances["sqrt_inv_matmul"])
        self.assertAllClose(inv_quad_res, inv_quad_actual, **self.tolerances["sqrt_inv_matmul"])

    def test_sqrt_inv_matmul_no_lhs(self):
        lazy_tensor = self.create_lazy_tensor().detach().requires_grad_(True)
        if len(lazy_tensor.batch_shape):
            return

        lazy_tensor_copy = lazy_tensor.clone().detach().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)

        # Create a test right hand side and left hand side
        rhs = torch.randn(*lazy_tensor.shape[:-1], 3).requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)

        # Perform forward pass
        sqrt_inv_matmul_res = lazy_tensor.sqrt_inv_matmul(rhs)
        evals, evecs = torch.linalg.eigh(evaluated)
        matrix_inv_root = evecs @ (evals.sqrt().reciprocal().unsqueeze(-1) * evecs.transpose(-1, -2))
        sqrt_inv_matmul_actual = matrix_inv_root @ rhs_copy

        # Check forward pass
        self.assertAllClose(sqrt_inv_matmul_res, sqrt_inv_matmul_actual, **self.tolerances["sqrt_inv_matmul"])

    def test_root_decomposition(self, cholesky=False):
        lazy_tensor = self.create_lazy_tensor()
        root_decomp = lazy_tensor.root_decomposition().root
        self.assertAllClose(root_decomp.evaluate(), torch.eye(lazy_tensor.size(-1)).expand(lazy_tensor.shape))

    def test_symeig(self):
        lazy_tensor = self.create_lazy_tensor()
        evals, evecs = lazy_tensor.symeig()
        self.assertAllClose(evals, torch.ones(lazy_tensor.shape[:-1]))
        self.assertAllClose(evecs.evaluate(), torch.eye(lazy_tensor.size(-1)).expand(lazy_tensor.shape))

    def test_svd(self):
        lazy_tensor = self.create_lazy_tensor()
        U, S, V = lazy_tensor.svd()
        self.assertAllClose(S, torch.ones(lazy_tensor.shape[:-1]))
        self.assertAllClose(U.evaluate(), torch.eye(lazy_tensor.size(-1)).expand(lazy_tensor.shape))
        self.assertAllClose(V.evaluate(), torch.eye(lazy_tensor.size(-1)).expand(lazy_tensor.shape))


class TestIdentityLazyTensorBatch(TestIdentityLazyTensor):
    def create_lazy_tensor(self):
        return IdentityLazyTensor(5, batch_shape=torch.Size([3, 6]))

    def evaluate_lazy_tensor(self, lazy_tensor):
        return torch.eye(5).expand(3, 6, 5, 5)


if __name__ == "__main__":
    unittest.main()
