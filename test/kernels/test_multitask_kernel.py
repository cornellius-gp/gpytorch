#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import LCMKernel, MultitaskKernel, RBFKernel, ScaleKernel


class TestMultitaskKernel(unittest.TestCase):
    def test_active_dims(self):
        # The data kernel only acts on its active dimensions.
        torch.manual_seed(0)
        x1, x2 = torch.rand(5, 3), torch.rand(4, 3)
        for data_covar_module in (RBFKernel(active_dims=[0, 2]), ScaleKernel(RBFKernel(active_dims=[0, 2]))):
            with self.subTest(data_covar_module=type(data_covar_module).__name__):
                kernel = MultitaskKernel(data_covar_module, num_tasks=2, rank=1)
                covar_i = kernel.task_covar_module.covar_matrix.to_dense()
                for diag in (False, True):
                    x2_ = x1 if diag else x2
                    covar_x = data_covar_module.forward(x1[:, [0, 2]], x2_[:, [0, 2]]).to_dense()
                    expected = torch.kron(covar_x, covar_i)
                    if diag:
                        expected = expected.diagonal()
                    res = kernel(x1, x2_, diag=diag)
                    self.assertTrue(torch.allclose(res.to_dense() if not diag else res, expected, atol=1e-6))

    def test_active_dims_lcm(self):
        # The base kernels of an LCMKernel only act on their active dimensions.
        torch.manual_seed(0)
        x1, x2 = torch.rand(5, 3), torch.rand(4, 3)
        kernel = LCMKernel([RBFKernel(active_dims=[0]), RBFKernel(active_dims=[1, 2])], num_tasks=2)
        expected = sum(
            torch.kron(
                k.data_covar_module.forward(x1[:, dims], x2[:, dims]).to_dense(),
                k.task_covar_module.covar_matrix.to_dense(),
            )
            for k, dims in zip(kernel.covar_module_list, ([0], [1, 2]))
        )
        self.assertTrue(torch.allclose(kernel(x1, x2).to_dense(), expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
