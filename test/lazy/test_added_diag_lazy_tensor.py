#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import AddedDiagLazyTensor, DiagLazyTensor, NonLazyTensor, RootLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestAddedDiagLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        tensor = torch.randn(5, 5)
        tensor = tensor.transpose(-1, -2).matmul(tensor).detach()
        diag = torch.tensor([1.0, 2.0, 4.0, 2.0, 3.0], requires_grad=True)
        return AddedDiagLazyTensor(NonLazyTensor(tensor), DiagLazyTensor(diag))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag_tensor._diag
        tensor = lazy_tensor._lazy_tensor.tensor
        return tensor + diag.diag()


class TestAddedDiagLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 4
    should_test_sample = True

    def create_lazy_tensor(self):
        tensor = torch.randn(3, 5, 5)
        tensor = tensor.transpose(-1, -2).matmul(tensor).detach()
        diag = torch.tensor(
            [[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]], requires_grad=True
        )
        return AddedDiagLazyTensor(NonLazyTensor(tensor), DiagLazyTensor(diag))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag_tensor._diag
        tensor = lazy_tensor._lazy_tensor.tensor
        return tensor + torch.diag_embed(diag, dim1=-2, dim2=-1)


class TestAddedDiagLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 4
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_lazy_tensor(self):
        tensor = torch.randn(4, 3, 5, 5)
        tensor = tensor.transpose(-1, -2).matmul(tensor).detach()
        diag = (
            torch.tensor(
                [[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]], requires_grad=True
            )
            .repeat(4, 1, 1)
            .detach()
        )
        return AddedDiagLazyTensor(NonLazyTensor(tensor), DiagLazyTensor(diag))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag_tensor._diag
        tensor = lazy_tensor._lazy_tensor.tensor
        return tensor + torch.diag_embed(diag, dim1=-2, dim2=-1)


class TestAddedDiagLazyTensorPrecondOverride(unittest.TestCase):
    def test_precond_solve(self):
        seed = 4
        torch.random.manual_seed(seed)

        tensor = torch.randn(1000, 800)
        diag = torch.abs(torch.randn(1000))

        standard_lt = AddedDiagLazyTensor(RootLazyTensor(tensor), DiagLazyTensor(diag))
        evals, evecs = standard_lt.symeig(eigenvectors=True)

        # this preconditioner is a simple example of near deflation
        def nonstandard_preconditioner(self):
            top_100_evecs = evecs[:, :100]
            top_100_evals = evals[:100] + 0.2 * torch.randn(100)

            precond_lt = RootLazyTensor(top_100_evecs @ torch.diag(top_100_evals ** 0.5))
            logdet = top_100_evals.log().sum()

            def precond_closure(rhs):
                rhs2 = top_100_evecs.t() @ rhs
                return top_100_evecs @ torch.diag(1.0 / top_100_evals) @ rhs2

            return precond_closure, precond_lt, logdet

        overrode_lt = AddedDiagLazyTensor(
            RootLazyTensor(tensor), DiagLazyTensor(diag), preconditioner_override=nonstandard_preconditioner
        )

        # compute a solve - mostly to make sure that we can actually perform the solve
        rhs = torch.randn(1000, 1)
        standard_solve = standard_lt.inv_matmul(rhs)
        overrode_solve = overrode_lt.inv_matmul(rhs)

        # gut checking that our preconditioner is not breaking anything
        self.assertEqual(standard_solve.shape, overrode_solve.shape)
        self.assertLess(torch.norm(standard_solve - overrode_solve) / standard_solve.norm(), 1.0)


if __name__ == "__main__":
    unittest.main()
