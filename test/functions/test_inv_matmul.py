#!/usr/bin/env python3

import unittest

import torch

from gpytorch import settings
from gpytorch.lazy import NonLazyTensor
from gpytorch.test.base_test_case import BaseTestCase


def _ensure_symmetric_grad(grad):
    """
    A gradient-hook hack to ensure that symmetric matrix gradients are symmetric
    """
    res = torch.add(grad, grad.transpose(-1, -2)).mul(0.5)
    return res


class TestInvMatmulNonBatch(BaseTestCase, unittest.TestCase):
    seed = 0

    def _create_mat(self):
        mat = torch.randn(8, 8)
        mat = mat @ mat.transpose(-1, -2)
        return mat

    def test_inv_matmul_vec(self):
        mat = self._create_mat().detach().requires_grad_(True)
        if mat.dim() > 2:  # This isn't a feature for batch mode
            return
        mat_copy = mat.detach().clone().requires_grad_(True)
        mat_copy.register_hook(_ensure_symmetric_grad)
        vec = torch.randn(mat.size(-1)).detach().requires_grad_(True)
        vec_copy = vec.detach().clone().requires_grad_(True)

        # Forward
        with settings.terminate_cg_by_size(False):
            res = NonLazyTensor(mat).inv_matmul(vec)
            actual = mat_copy.inverse().matmul(vec_copy)
            self.assertAllClose(res, actual)

            # Backward
            grad_output = torch.randn_like(vec)
            res.backward(gradient=grad_output)
            actual.backward(gradient=grad_output)
            self.assertAllClose(mat.grad, mat_copy.grad)
            self.assertAllClose(vec.grad, vec_copy.grad)

    def test_inv_matmul_multiple_vecs(self):
        mat = self._create_mat().detach().requires_grad_(True)
        mat_copy = mat.detach().clone().requires_grad_(True)
        mat_copy.register_hook(_ensure_symmetric_grad)
        vecs = torch.randn(*mat.shape[:-2], mat.size(-1), 4).detach().requires_grad_(True)
        vecs_copy = vecs.detach().clone().requires_grad_(True)

        # Forward
        with settings.terminate_cg_by_size(False):
            res = NonLazyTensor(mat).inv_matmul(vecs)
            actual = mat_copy.inverse().matmul(vecs_copy)
            self.assertAllClose(res, actual)

            # Backward
            grad_output = torch.randn_like(vecs)
            res.backward(gradient=grad_output)
            actual.backward(gradient=grad_output)
            self.assertAllClose(mat.grad, mat_copy.grad)
            self.assertAllClose(vecs.grad, vecs_copy.grad)


class TestInvMatmulBatch(TestInvMatmulNonBatch):
    seed = 0

    def _create_mat(self):
        mats = torch.randn(2, 8, 8)
        mats = mats @ mats.transpose(-1, -2)
        return mats


if __name__ == "__main__":
    unittest.main()
