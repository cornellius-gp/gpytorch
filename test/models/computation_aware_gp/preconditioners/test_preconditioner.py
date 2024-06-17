#!/usr/bin/env python3

import abc
import unittest
from abc import abstractmethod

import torch

import gpytorch
from gpytorch import kernels
from gpytorch.models.computation_aware_iterative_gp import preconditioners

N_PTS = 100


class BasePreconditionerTestCase(abc.ABC):
    def create_kernel(self):
        return kernels.MaternKernel(nu=2.5)

    def create_train_data(self):
        return torch.randn(N_PTS, 1)

    def create_likelihood(self):
        return gpytorch.likelihoods.GaussianLikelihood()

    def create_test_data(self):
        return torch.randn(10, 1)

    @abstractmethod
    def create_preconditioner(self) -> preconditioners.Preconditioner:
        raise NotImplementedError()

    def test_sqrt_inv_matmul_shape(self):
        preconditioner = self.create_preconditioner()
        mat = self.create_kernel()(self.create_train_data(), self.create_test_data()).to_dense()
        result = preconditioner.sqrt_inv_matmul(mat)
        self.assertEqual(result.shape, (preconditioner.shape[-2], mat.shape[-1]))

    def test_inv_matmul_shape(self):
        preconditioner = self.create_preconditioner()
        mat = self.create_kernel()(self.create_train_data(), self.create_test_data()).to_dense()
        result_mat = preconditioner.inv_matmul(mat)
        self.assertEqual(result_mat.shape, (preconditioner.shape[-2], mat.shape[-1]))
        vec = mat[:, 0]
        result_vec = preconditioner.inv_matmul(vec)
        self.assertEqual(result_vec.shape, (preconditioner.shape[-2],))


if __name__ == "__main__":
    unittest.main()
