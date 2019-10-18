import torch
import matplotlib.pyplot as plt
import math
import numpy as np

from gpytorch.kernels import RBFKernel, SpectralGPKernel

import unittest

class TestSpectralGPKernel(unittest.TestCase):
    def test_trapezoidal_integration(self, n = 2500):
        print("test trap")
        #omega = torch.linspace(0, 1./0.1, n)
        tau = np.arange(30)
        kernel = SpectralGPKernel(torch.from_numpy(tau).float())
        omega = kernel.omega

        # check that the generated omegas are computed properly
        #self.assertEqual((omega - kernel.omega.squeeze()).norm().item(), 0.)
        #tau = np.arange(30)

        density = torch.randn_like(omega).abs() / (omega + 1)
        density = density / np.trapz(density.numpy(), omega.numpy())

        numpy_integral = np.zeros(30)
        for ii in range(30):
            numpy_integral[ii] = np.trapz(density.numpy() * np.cos(2.0 * np.pi * tau[ii] * omega.numpy()), omega.numpy()) #/ (2*np.pi)

        vector_numpy = np.trapz(density.numpy() * np.cos(2.0 * np.pi * tau.reshape(-1,1) * omega.view(1,-1).numpy()), omega.numpy() ) #/ (2*np.pi)
        #print(numpy_integral - vector_numpy)
        torch_integral = kernel.compute_kernel_values(torch.tensor(tau).float().view(1,-1), density)

        self.assertLess(np.linalg.norm(torch_integral.view(-1).numpy() - numpy_integral), 2e-5)


    def test_kernel(self, seed=1, n=500):
        print("test kernel")
        torch.random.manual_seed(seed)
        # input data
        x = torch.linspace(-50, 50, 201).float().view(-1,1)

        # generating kernel values
        kernel = RBFKernel()
        kernel._set_lengthscale(3.)
        k = kernel(x, torch.zeros(1,1)).evaluate()
        k = k.detach()

        sgp_kernel = SpectralGPKernel(x, num_locs=50)
        # two times the psd because we are only considering half of it
        true_spectral_density = 2 * (2.0 * math.pi * kernel.lengthscale**2)**0.5 * \
                    torch.exp(-2. * math.pi**2 * kernel.lengthscale**2 * sgp_kernel.omega**2)
        kernel_output = sgp_kernel.compute_kernel_values(x, true_spectral_density)

        #print('norm of reconstruction: ', (k - kernel_output).norm())
        self.assertLess((k - kernel_output).norm(), 1e-5)

    ## TODO: write test for initializing model

if __name__ == "__main__":
    unittest.main()
