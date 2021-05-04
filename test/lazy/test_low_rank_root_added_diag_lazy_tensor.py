#!/usr/bin/env python3

import math
import unittest
from unittest.mock import MagicMock, patch

import torch

import gpytorch
from gpytorch.lazy import LowRankRootAddedDiagLazyTensor, LowRankRootLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase, _ensure_symmetric_grad


class TestLowRankRootAddedDiagLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        tensor = torch.randn(5, 2)
        diag = torch.tensor([1.0, 2.0, 4.0, 2.0, 3.0])
        lt = LowRankRootLazyTensor(tensor).add_diag(diag)
        assert isinstance(lt, LowRankRootAddedDiagLazyTensor)
        return lt

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag_tensor._diag
        root = lazy_tensor._lazy_tensor.root.tensor
        return root @ root.transpose(-1, -2) + diag.diag_embed(dim1=-2, dim2=-1)

    def _test_inv_matmul(self, rhs, lhs=None, cholesky=False):
        lazy_tensor = self.create_lazy_tensor().requires_grad_(True)
        lazy_tensor_copy = lazy_tensor.clone().detach_().requires_grad_(True)
        evaluated = self.evaluate_lazy_tensor(lazy_tensor_copy)
        evaluated.register_hook(_ensure_symmetric_grad)

        # Create a test right hand side and left hand side
        rhs.requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)
        if lhs is not None:
            lhs.requires_grad_(True)
            lhs_copy = lhs.clone().detach().requires_grad_(True)

        _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
        with patch("gpytorch.utils.linear_cg", new=_wrapped_cg) as linear_cg_mock:
            with gpytorch.settings.max_cholesky_size(math.inf if cholesky else 0), gpytorch.settings.cg_tolerance(1e-4):
                # Perform the inv_matmul
                if lhs is not None:
                    res = lazy_tensor.inv_matmul(rhs, lhs)
                    actual = lhs_copy @ evaluated.inverse() @ rhs_copy
                else:
                    res = lazy_tensor.inv_matmul(rhs)
                    actual = evaluated.inverse().matmul(rhs_copy)
                self.assertAllClose(res, actual, rtol=0.02, atol=1e-5)

                # Perform backward pass
                grad = torch.randn_like(res)
                res.backward(gradient=grad)
                actual.backward(gradient=grad)
                for arg, arg_copy in zip(lazy_tensor.representation(), lazy_tensor_copy.representation()):
                    if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                        self.assertAllClose(arg.grad, arg_copy.grad, rtol=0.03, atol=1e-5)
                self.assertAllClose(rhs.grad, rhs_copy.grad, rtol=0.03, atol=1e-5)
                if lhs is not None:
                    self.assertAllClose(lhs.grad, lhs_copy.grad, rtol=0.03, atol=1e-5)

            self.assertFalse(linear_cg_mock.called)

    def _test_inv_quad_logdet(self, reduce_inv_quad=True, cholesky=False):
        if not self.__class__.skip_slq_tests:
            # Forward
            lazy_tensor = self.create_lazy_tensor()
            evaluated = self.evaluate_lazy_tensor(lazy_tensor)
            flattened_evaluated = evaluated.view(-1, *lazy_tensor.matrix_shape)

            vecs = torch.randn(*lazy_tensor.batch_shape, lazy_tensor.size(-1), 3, requires_grad=True)
            vecs_copy = vecs.clone().detach_().requires_grad_(True)

            _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
            with patch("gpytorch.utils.linear_cg", new=_wrapped_cg) as linear_cg_mock:
                with gpytorch.settings.num_trace_samples(256), gpytorch.settings.max_cholesky_size(
                    math.inf if cholesky else 0
                ), gpytorch.settings.cg_tolerance(1e-5):

                    res_inv_quad, res_logdet = lazy_tensor.inv_quad_logdet(
                        inv_quad_rhs=vecs, logdet=True, reduce_inv_quad=reduce_inv_quad
                    )

            actual_inv_quad = evaluated.inverse().matmul(vecs_copy).mul(vecs_copy).sum(-2)
            if reduce_inv_quad:
                actual_inv_quad = actual_inv_quad.sum(-1)
            actual_logdet = torch.cat(
                [torch.logdet(flattened_evaluated[i]).unsqueeze(0) for i in range(lazy_tensor.batch_shape.numel())]
            ).view(lazy_tensor.batch_shape)

            self.assertAllClose(res_inv_quad, actual_inv_quad, rtol=0.01, atol=0.01)
            self.assertAllClose(res_logdet, actual_logdet, rtol=0.2, atol=0.03)

            self.assertFalse(linear_cg_mock.called)

    def test_root_decomposition_cholesky(self):
        self.test_root_decomposition(cholesky=True)


class TestLowRankRootAddedDiagLazyTensorBatch(TestLowRankRootAddedDiagLazyTensor):
    seed = 4
    should_test_sample = True

    def create_lazy_tensor(self):
        tensor = torch.randn(3, 5, 2)
        diag = torch.tensor([[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]])
        lt = LowRankRootLazyTensor(tensor).add_diag(diag)
        assert isinstance(lt, LowRankRootAddedDiagLazyTensor)
        return lt

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag_tensor._diag
        root = lazy_tensor._lazy_tensor.root.tensor
        return root @ root.transpose(-1, -2) + diag.diag_embed(dim1=-2, dim2=-1)


class TestLowRankRootAddedDiagLazyTensorMultiBatch(TestLowRankRootAddedDiagLazyTensor):
    seed = 4
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_lazy_tensor(self):
        tensor = torch.randn(4, 3, 5, 2)
        diag = torch.tensor([[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]]).repeat(
            4, 1, 1
        )
        lt = LowRankRootLazyTensor(tensor).add_diag(diag)
        assert isinstance(lt, LowRankRootAddedDiagLazyTensor)
        return lt

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag_tensor._diag
        root = lazy_tensor._lazy_tensor.root.tensor
        return root @ root.transpose(-1, -2) + diag.diag_embed(dim1=-2, dim2=-1)


if __name__ == "__main__":
    unittest.main()
