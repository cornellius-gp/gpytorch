import torch
import matplotlib.pyplot as plt
import math
import numpy as np

from gpytorch.kernels import RBFKernel

from spectralgp.kernels import SpectralGPKernel

import unittest

class SpectralGPKernelTest(unittest.TestCase):
    def test_trapezoidal_integration(self, n = 2500):
        print("test trap")
        #omega = torch.linspace(start=0, end=math.pi-width/2, steps=n)
        omega = torch.linspace(0, 1./0.1, n)

        kernel = SpectralGPKernel(integration='U', num_locs = n, omega_max=1./0.1)

        # check that the generated omegas are computed properly
        self.assertEqual((omega - kernel.omega.squeeze()).norm().item(), 0.)
        tau = np.arange(30)

        density = torch.randn_like(omega).abs() / (omega + 1)
        density = density / np.trapz(density.numpy(), omega.numpy())

        numpy_integral = np.zeros(30)
        for ii in range(30):
            numpy_integral[ii] = np.trapz(density.numpy() * np.cos(2.0 * np.pi * tau[ii] * omega.numpy()), omega.numpy()) #/ (2*np.pi)

        vector_numpy = np.trapz(density.numpy() * np.cos(2.0 * np.pi * tau.reshape(-1,1) * omega.view(1,-1).numpy()), omega.numpy() ) #/ (2*np.pi)
        #print(numpy_integral - vector_numpy)
        torch_integral = kernel.compute_kernel_values(torch.tensor(tau).float().view(1,-1), density)

        # plt.plot(tau, torch_integral.t().numpy(), label='Integration Output')
        # plt.plot(tau, numpy_integral, label='True K')
        # #plt.plot(tau, torch_integral.view(-1).numpy() / numpy_integral)
        # plt.legend()
        # plt.xlabel('x')
        # plt.ylabel('K(x,0)')
        # plt.title("Test Trap Output")
        # plt.show()

        #print(torch_integral.size(), numpy_integral.shape)
        self.assertLess(np.linalg.norm(torch_integral.view(-1).numpy() - numpy_integral), 2e-5)

    def test_mc_integration(self, n = 2500, seed = 1):
        print("test MC")
        torch.random.manual_seed(seed)

        # now perform a check of the monte carlo estimation
        kernel_mc = SpectralGPKernel(integration='MC', num_locs = n)

        omega = kernel_mc.omega

        tau = np.arange(30)
        density = torch.randn_like(omega).abs() / (omega + 1)

        numpy_integral = np.zeros(30)
        for ii in range(30):
            numpy_integral[ii] = np.mean( density.numpy() * np.cos(2.0 * math.pi * tau[ii] * omega.numpy()) )# / (2 *np.pi)

        torch_integral = kernel_mc.compute_kernel_values(torch.tensor(tau).float().view(1,-1), density, integration='MC')

        # plt.plot(tau, torch_integral.squeeze().numpy(), label='Integration Output')
        # plt.plot(tau, numpy_integral, label='True K')
        # plt.legend()
        # plt.xlabel('x')
        # plt.ylabel('K(x,0)')
        # plt.title("Test MC Output")
        # plt.show()

        self.assertLess(np.linalg.norm(torch_integral.view(-1).numpy() - numpy_integral), 1e-3)

    def test_kernel(self, seed=1, n=500):
        print("test kernel")
        torch.random.manual_seed(seed)
        # input data
        x = torch.linspace(-10, 10, 201).view(-1,1)

        nf = 2.0 * torch.mean(x[1:] - x[:-1])
        print('nyquist frequency: ', nf)
        # generating kernel values
        kernel = RBFKernel()
        kernel._set_lengthscale(3.)
        k = kernel(x, torch.zeros(1,1)).evaluate()
        k = k.detach()

        # extracting approximate spectral density
        omega = torch.linspace(0, 1./nf, n)
        tau = x.data

        s = torch.zeros(n)
        for ii in range(n):
            s[ii] = torch.dot(k.data.squeeze(), torch.cos(2 * math.pi * tau.squeeze() * omega[ii]))

        # reconstructing kernel using mean
        kernel_rec = SpectralGPKernel(integration='U', num_locs = n, omega_max=1./nf)

        # check that the generated omegas are computed properly
        self.assertEqual((omega - kernel_rec.omega.squeeze()).norm().item(), 0.)

        kernel_output = kernel_rec.compute_kernel_values(tau, s, integration='U').view(-1)

        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # ax[0].plot(x.numpy(), kernel_output.numpy(), label='Integration Output')
        # ax[0].plot(x.numpy(), k.numpy(), label='True K')
        # ax[0].legend()
        # ax[0].set_xlabel('x')
        # ax[0].set_ylabel('K(x,0)')

        # true_s = (2*math.pi*kernel.lengthscale**2)**(0.5) * torch.exp(-2*(math.pi * kernel.lengthscale * omega)**2)
        # ax[1].plot(omega.numpy(), s.numpy() * (0.5 * nf.numpy()), label = 'DTFT')
        # ax[1].plot(omega.numpy(), true_s.squeeze(0).detach().numpy(), label = 'True')
        # ax[1].legend()
        # ax[1].set_xlabel('omega')
        # ax[1].set_ylabel('S(omega)')
        # plt.show()

        relative_error = (kernel_output.view(-1) - k.view(-1)).norm() / k.norm()
        print('relative error: ', relative_error)
        self.assertLess(relative_error, 1e-3)

    def test_kernel_forwards(self, seed=1, n = 500):
        print("test kernel")
        torch.random.manual_seed(seed)
        # input data
        x = torch.linspace(-100, 100, 201).view(-1,1)

        # generating kernel values
        kernel = RBFKernel()
        kernel._set_lengthscale(3.)
        k = kernel(x, torch.zeros(1,1)).evaluate()
        k = k.detach()

        # reconstructing kernel using mean
        kernel_rec = SpectralGPKernel(integration='U', num_locs= n)

        # with torch.no_grad():
        #     kern_output = kernel_rec(x, torch.zeros(1)).evaluate()

        #     # random sample from prior kernel
        #     fig, ax = plt.subplots(nrows=1, ncols=1)

        #     plt.plot(x.numpy(), kern_output.numpy(), label = 'Prior')
        #     plt.plot(x.numpy(), k.numpy(), label = 'RBF')
        #     plt.xlabel('x')
        #     plt.ylabel('K(x)')
        #     plt.legend()
        #     plt.show()

if __name__ == "__main__":
    unittest.main()
