import torch
#import matplotlib.pyplot as plt
import math
import numpy as np
import gpytorch
from gpytorch.kernels import RBFKernel, SpectralGPKernel, WrappedSpectralGPKernel

import unittest

class WrappedSpectralGPKernelTest(unittest.TestCase):
    def test_forward(self):
        print("test fwd")

        # dummy data #
        grid_bounds = [(0, 5), (0, 5)]
        grid_size = 25
        grid = torch.zeros(grid_size, len(grid_bounds))
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff,
                                        grid_bounds[i][1] + grid_diff, grid_size)

        train_x = gpytorch.utils.grid.create_data_from_grid(grid)
        train_y = 2. * torch.sin((train_x[:, 0] * math.pi))

        ## dists for mixture of gaussian spectral densities ##
        dist1 = torch.distributions.Normal(1., 0.2)
        dist2 = torch.distributions.Normal(2., 0.2)

        ## wrapped FKL and set latents ##
        wrapped_fkl = WrappedSpectralGPKernel(train_x, train_y, shared=False)

        omg1 = wrapped_fkl.get_omega(0)
        dens1 = dist1.log_prob(omg1)
        wrapped_fkl.set_latent_params(dens1, 0)
        wrapped_cov1 = wrapped_fkl.base_kernels[0](train_x[:, 0], train_x[:, 0])

        omg2 = wrapped_fkl.get_omega(1)
        dens2 = dist2.log_prob(omg2)
        wrapped_fkl.set_latent_params(dens2, 1)
        wrapped_cov2 = wrapped_fkl.base_kernels[1](train_x[:, 1], train_x[:, 1])

        wrapped_fkl_cov = wrapped_fkl(train_x, train_x)

        ## now use two independent spectral GP models ##
        kern1 = SpectralGPKernel(omega=omg1)
        kern1.set_latent_params(dens1)

        kern2 = SpectralGPKernel(omega=omg2)
        kern2.set_latent_params(dens2)

        cov1 = kern1(train_x[:, 0], train_x[:, 0])
        cov2 = kern2(train_x[:, 1], train_x[:, 1])

        test_cov = cov1 * cov2

        self.assertLess((cov1.evaluate() - wrapped_cov1.evaluate()).norm().abs(), 1e-6)
        self.assertLess((cov2.evaluate() - wrapped_cov2.evaluate()).norm().abs(), 1e-6)
        print("Individual matrices good")
        self.assertLess((test_cov.evaluate().norm() - wrapped_fkl_cov.evaluate().norm()).abs(), 1e-4)

    # def test_subkernels(self):
    #     print("test subkernels")
    #
    #     # dummy data #
    #     grid_bounds = [(0, 5), (0, 5)]
    #     grid_size = 25
    #     grid = torch.zeros(grid_size, len(grid_bounds))
    #     for i in range(len(grid_bounds)):
    #         grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
    #         grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff,
    #                                     grid_bounds[i][1] + grid_diff, grid_size)
    #
    #     train_x = gpytorch.utils.grid.create_data_from_grid(grid)
    #     train_y = 2. * torch.sin((train_x[:, 0] * math.pi))
    #
    #     ## dists for mixture of gaussian spectral densities ##
    #     dist1 = torch.distributions.Normal(1., 0.2)
    #     dist2 = torch.distributions.Normal(2., 0.2)
    #
    #     ## wrapped FKL and set latents ##
    #     wrapped_fkl = WrappedSpectralGPKernel(train_x, train_y, shared=True)
    #
    #     omg1 = wrapped_fkl.get_omega(0)
    #     dens1 = dist1.log_prob(omg1)
    #     wrapped_fkl.set_latent_params(dens1, 0)
    #     wrapped_cov1 = wrapped_fkl.base_kernels[0](train_x[:, 0], train_x[:, 0])
    #
    #     omg2 = wrapped_fkl.get_omega(1)
    #     dens2 = dist2.log_prob(omg2)
    #     wrapped_fkl.set_latent_params(dens2, 1)
    #     wrapped_cov2 = wrapped_fkl.base_kernels[1](train_x[:, 1], train_x[:, 1])
    #
    #     ## now use two independent spectral GP models ##
    #     kern1 = SpectralGPKernel(omega=omg1)
    #     kern1.set_latent_params(dens1)
    #
    #     kern2 = SpectralGPKernel(omega=omg2)
    #     kern2.set_latent_params(dens2)
    #
    #     cov1 = kern1(train_x[:, 0], train_x[:, 0])
    #     cov2 = kern2(train_x[:, 1], train_x[:, 1])
    #
    #     self.assertLess((cov1.evaluate() - wrapped_cov1.evaluate()).norm().abs(), 1e-6)
    #     self.assertLess((cov2.evaluate() - wrapped_cov2.evaluate()).norm().abs(), 1e-6)


if __name__ == "__main__":
    unittest.main()
