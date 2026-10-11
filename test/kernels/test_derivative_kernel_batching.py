#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import Matern52KernelGrad, RBFKernelGrad, RBFKernelGradGrad


class TestDerivativeKernelBatching(unittest.TestCase):
    def test_broadcast_batches(self):
        for kernel_class in (RBFKernelGrad, RBFKernelGradGrad, Matern52KernelGrad):
            for input_batch, parameter_batch in (((), (3,)), ((), (2, 3)), ((2, 1), (3,))):
                with self.subTest(kernel=kernel_class.__name__, inputs=input_batch, parameters=parameter_batch):
                    x1 = torch.randn(*input_batch, 3, 2, dtype=torch.double)
                    x2 = torch.randn(3, 2, dtype=torch.double)
                    kernel = kernel_class(ard_num_dims=2, batch_shape=torch.Size(parameter_batch)).double()
                    lengthscales = torch.linspace(0.5, 1.5, kernel.raw_lengthscale.numel(), dtype=torch.double)
                    kernel.lengthscale = lengthscales.reshape(kernel.raw_lengthscale.shape)
                    batch = torch.broadcast_shapes(input_batch, parameter_batch)
                    first = x1.expand(*batch, 3, 2).reshape(-1, 3, 2)
                    second = x2.expand(*batch, 3, 2).reshape(-1, 3, 2)
                    scales = kernel.lengthscale.expand(*batch, 1, 2).reshape(-1, 1, 2)
                    covariances, diagonals = [], []
                    for a, b, scale in zip(first, second, scales):
                        reference = kernel_class(ard_num_dims=2).double()
                        reference.lengthscale = scale
                        covariances.append(reference(a, b).to_dense())
                        diagonals.append(reference(a, diag=True))
                    expected = torch.stack(covariances).reshape(*batch, *covariances[0].shape)
                    expected_diag = torch.stack(diagonals).reshape(*batch, -1)
                    torch.testing.assert_close(kernel(x1, x2).to_dense(), expected)
                    torch.testing.assert_close(kernel(x1, diag=True), expected_diag)


if __name__ == "__main__":
    unittest.main()
