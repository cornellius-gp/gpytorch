#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import MultitaskKernel, RBFKernel


class TestMultitaskBatch(unittest.TestCase):
    def test_existing_task_batches_are_broadcast(self):
        for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            kernel = MultitaskKernel(
                RBFKernel(batch_shape=torch.Size([2])),
                num_tasks=2,
                batch_shape=torch.Size([2]),
            ).to(device=device, dtype=torch.double)
            for batch in [(), (2,), (3, 2)]:
                with self.subTest(device=device, batch=batch):
                    x1 = torch.rand(*batch, 4, 1, device=device, dtype=torch.double)
                    x2 = torch.rand(*batch, 3, 1, device=device, dtype=torch.double)
                    data = kernel.data_covar_module(x1, x2).to_dense()
                    task = kernel.task_covar_module.covar_matrix.to_dense()
                    expected = torch.einsum("...ij,...kl->...ikjl", data, task)
                    expected = expected.reshape(*data.shape[:-2], 8, 6)
                    actual = kernel(x1, x2).to_dense()
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertTrue(torch.allclose(actual, expected))
                    full = kernel(x1).to_dense()
                    diagonal = kernel(x1, diag=True)
                    self.assertTrue(torch.allclose(diagonal, full.diagonal(dim1=-1, dim2=-2)))
