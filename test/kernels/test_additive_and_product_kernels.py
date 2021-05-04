#!/usr/bin/env python3

import math
import unittest

import torch

import gpytorch
from gpytorch.kernels import LinearKernel, MaternKernel, RBFKernel, RFFKernel


class TestModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            RFFKernel(active_dims=[0], num_samples=10) + MaternKernel(nu=2.5, active_dims=[1, 2])
        )

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class TestModelNoStructure(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            RBFKernel(active_dims=[0], num_samples=10) + MaternKernel(nu=2.5, active_dims=[1, 2])
        )

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class TestAdditiveAndProductKernel(unittest.TestCase):
    def test_computes_product_of_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = kernel_1 * kernel_2

        actual = torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float)
        actual = actual.mul_(-0.5).div_(lengthscale ** 2).exp() ** 2

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

    def test_computes_sum_of_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = kernel_1 + kernel_2

        actual = torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float)
        actual = actual.mul_(-0.5).div_(lengthscale ** 2).exp() * 2

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

    def test_computes_sum_of_radial_basis_function_diag(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)

        actual = torch.tensor([0.2702, 2.000, 0.0222])

        lengthscale = 2.0

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = kernel_1 + kernel_2
        kernel.eval()

        kernel.eval()
        res = kernel(a, b, diag=True)
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_computes_sum_of_three_radial_basis_function_diag(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)

        actual = torch.tensor([0.4060, 3.000, 0.0333])

        lengthscale = 2.0

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_3 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = kernel_1 + kernel_2 + kernel_3
        kernel.eval()

        res = kernel(a, b, diag=True)
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_computes_product_of_radial_basis_function_diag(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)

        actual = torch.tensor([2.4788e-03, 1.000, 1.3710e-06])

        lengthscale = 2.0

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_3 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = kernel_1 * kernel_2 * kernel_3
        kernel.eval()

        kernel.eval()
        res = kernel(a, b, diag=True)
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_computes_product_of_three_radial_basis_function_diag(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)

        actual = torch.tensor([1.8316e-02, 1.000, 1.2341e-04])

        lengthscale = 2.0

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = kernel_1 * kernel_2
        kernel.eval()

        kernel.eval()
        res = kernel(a, b, diag=True)
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_computes_product_of_three_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_3 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = (kernel_1 * kernel_2) * kernel_3
        self.assertEqual(len(kernel.kernels), 3)
        for sub_kernel in kernel.kernels:
            self.assertIsInstance(sub_kernel, RBFKernel)

        actual = torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float)
        actual = actual.mul_(-0.5).div_(lengthscale ** 2).exp() ** 3

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_3 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = (kernel_1 * kernel_2) * kernel_3
        self.assertEqual(len(kernel.kernels), 3)
        for sub_kernel in kernel.kernels:
            self.assertIsInstance(sub_kernel, RBFKernel)

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

    def test_computes_product_of_four_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_3 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_4 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = kernel_1 * kernel_2 * kernel_3 * kernel_4
        self.assertEqual(len(kernel.kernels), 4)
        for sub_kernel in kernel.kernels:
            self.assertIsInstance(sub_kernel, RBFKernel)

        actual = torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float)
        actual = actual.mul_(-0.5).div_(lengthscale ** 2).exp() ** 4

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

    def test_computes_sum_of_four_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_3 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_4 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = (kernel_1 + kernel_2) + (kernel_3 + kernel_4)
        self.assertEqual(len(kernel.kernels), 4)
        for sub_kernel in kernel.kernels:
            self.assertIsInstance(sub_kernel, RBFKernel)

        actual = (
            torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float).mul_(-0.5).div_(lengthscale ** 2).exp() * 4
        )

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

    def test_computes_sum_of_three_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_3 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = (kernel_1 + kernel_2) + kernel_3
        self.assertEqual(len(kernel.kernels), 3)
        for sub_kernel in kernel.kernels:
            self.assertIsInstance(sub_kernel, RBFKernel)

        actual = (
            torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float).mul_(-0.5).div_(lengthscale ** 2).exp() * 3
        )

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_3 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = kernel_1 + (kernel_2 + kernel_3)
        self.assertEqual(len(kernel.kernels), 3)
        for sub_kernel in kernel.kernels:
            self.assertIsInstance(sub_kernel, RBFKernel)

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

    def test_computes_sum_radial_basis_function_gradient(self):
        softplus = torch.nn.functional.softplus
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)
        lengthscale = 2

        param = math.log(math.exp(lengthscale) - 1) * torch.ones(3, 3)
        param.requires_grad_()
        diffs = a.expand(3, 3) - b.expand(3, 3).transpose(0, 1)
        actual_output = (-0.5 * (diffs / softplus(param)) ** 2).exp()
        actual_output.backward(torch.eye(3))
        actual_param_grad = param.grad.sum() * 2

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = kernel_1 + kernel_2
        kernel.eval()

        output = kernel(a, b).evaluate()
        output.backward(gradient=torch.eye(3))
        res = kernel.kernels[0].raw_lengthscale.grad + kernel.kernels[1].raw_lengthscale.grad
        self.assertLess(torch.norm(res - actual_param_grad), 2e-5)

    def test_computes_sum_three_radial_basis_function_gradient(self):
        softplus = torch.nn.functional.softplus
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)
        lengthscale = 2

        param = math.log(math.exp(lengthscale) - 1) * torch.ones(3, 3)
        param.requires_grad_()
        diffs = a.expand(3, 3) - b.expand(3, 3).transpose(0, 1)
        actual_output = (-0.5 * (diffs / softplus(param)) ** 2).exp()
        actual_output.backward(torch.eye(3))
        actual_param_grad = param.grad.sum() * 3

        kernel_1 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_2 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel_3 = RBFKernel().initialize(lengthscale=lengthscale)
        kernel = kernel_1 + kernel_2 + kernel_3
        kernel.eval()

        output = kernel(a, b).evaluate()
        output.backward(gradient=torch.eye(3))
        res = (
            kernel.kernels[0].raw_lengthscale.grad
            + kernel.kernels[1].raw_lengthscale.grad
            + kernel.kernels[2].raw_lengthscale.grad
        )
        self.assertLess(torch.norm(res - actual_param_grad), 2e-5)

    def test_is_stationary(self):
        kernel_1 = RBFKernel().initialize(lengthscale=1)
        kernel_2 = RBFKernel().initialize(lengthscale=2)
        kernel_3 = LinearKernel().initialize()

        self.assertTrue((kernel_1 + kernel_2).is_stationary)
        self.assertTrue((kernel_1 * kernel_2).is_stationary)
        self.assertFalse((kernel_1 + kernel_3).is_stationary)
        self.assertFalse((kernel_1 * kernel_3).is_stationary)

    def test_kernel_output(self):
        train_x = torch.randn(1000, 3)
        train_y = torch.randn(1000)
        model = TestModel(train_x, train_y)

        # Make sure that the prior kernel is the correct type
        model.train()
        output = model(train_x).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, gpytorch.lazy.SumLazyTensor)

        # Make sure that the prior predictive kernel is the correct type
        model.train()
        output = model.likelihood(model(train_x)).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, gpytorch.lazy.AddedDiagLazyTensor)

    def test_kernel_output_no_structure(self):
        train_x = torch.randn(1000, 3)
        train_y = torch.randn(1000)
        model = TestModelNoStructure(train_x, train_y)

        # Make sure that the prior kernel is the correct type
        model.train()
        output = model(train_x).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, gpytorch.lazy.ConstantMulLazyTensor)

        # Make sure that the prior predictive kernel is the correct type
        model.train()
        output = model.likelihood(model(train_x)).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, gpytorch.lazy.AddedDiagLazyTensor)


if __name__ == "__main__":
    unittest.main()
