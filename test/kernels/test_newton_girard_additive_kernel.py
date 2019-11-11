#!/usr/bin/env python3

from unittest import TestCase

import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import AdditiveKernel, NewtonGirardAdditiveKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestNewtonGirardAdditiveKernel(TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return NewtonGirardAdditiveKernel(RBFKernel(), 4, 2, **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return NewtonGirardAdditiveKernel(RBFKernel(ard_num_dims=num_dims), num_dims, 2, **kwargs)

    def test_degree1(self):
        AddK = NewtonGirardAdditiveKernel(RBFKernel(ard_num_dims=3), 3, 1)
        self.assertEqual(AddK.base_kernel.lengthscale.numel(), 3)
        self.assertEqual(AddK.outputscale.numel(), 1)

        testvals = torch.tensor([[1, 2, 3], [7, 5, 2]], dtype=torch.float)
        add_k_val = AddK(testvals, testvals).evaluate()

        manual_k = ScaleKernel(
            AdditiveKernel(RBFKernel(active_dims=0), RBFKernel(active_dims=1), RBFKernel(active_dims=2))
        )
        manual_k.initialize(outputscale=1.0)
        manual_add_k_val = manual_k(testvals, testvals).evaluate()

        # np.testing.assert_allclose(add_k_val.detach().numpy(), manual_add_k_val.detach().numpy(), atol=1e-5)
        self.assertTrue(torch.allclose(add_k_val, manual_add_k_val, atol=1e-5))

    def test_degree2(self):
        AddK = NewtonGirardAdditiveKernel(RBFKernel(ard_num_dims=3), 3, 2)
        self.assertEqual(AddK.base_kernel.lengthscale.numel(), 3)
        self.assertEqual(AddK.outputscale.numel(), 2)

        testvals = torch.tensor([[1, 2, 3], [7, 5, 2]], dtype=torch.float)
        add_k_val = AddK(testvals, testvals).evaluate()

        manual_k1 = ScaleKernel(
            AdditiveKernel(RBFKernel(active_dims=0), RBFKernel(active_dims=1), RBFKernel(active_dims=2))
        )
        manual_k1.initialize(outputscale=1 / 2)
        manual_k2 = ScaleKernel(
            AdditiveKernel(RBFKernel(active_dims=[0, 1]), RBFKernel(active_dims=[1, 2]), RBFKernel(active_dims=[0, 2]))
        )
        manual_k2.initialize(outputscale=1 / 2)
        manual_k = AdditiveKernel(manual_k1, manual_k2)
        manual_add_k_val = manual_k(testvals, testvals).evaluate()

        # np.testing.assert_allclose(add_k_val.detach().numpy(), manual_add_k_val.detach().numpy(), atol=1e-5)
        self.assertTrue(torch.allclose(add_k_val, manual_add_k_val, atol=1e-5))

    def test_degree3(self):
        # just make sure it doesn't break here.
        AddK = NewtonGirardAdditiveKernel(RBFKernel(ard_num_dims=3), 3, 3)
        self.assertEqual(AddK.base_kernel.lengthscale.numel(), 3)
        self.assertEqual(AddK.outputscale.numel(), 3)

        testvals = torch.tensor([[1, 2, 3], [7, 5, 2]], dtype=torch.float)
        add_k_val = AddK(testvals, testvals).evaluate()

        manual_k1 = ScaleKernel(
            AdditiveKernel(RBFKernel(active_dims=0), RBFKernel(active_dims=1), RBFKernel(active_dims=2))
        )
        manual_k1.initialize(outputscale=1 / 3)
        manual_k2 = ScaleKernel(
            AdditiveKernel(RBFKernel(active_dims=[0, 1]), RBFKernel(active_dims=[1, 2]), RBFKernel(active_dims=[0, 2]))
        )
        manual_k2.initialize(outputscale=1 / 3)

        manual_k3 = ScaleKernel(AdditiveKernel(RBFKernel()))
        manual_k3.initialize(outputscale=1 / 3)
        manual_k = AdditiveKernel(manual_k1, manual_k2, manual_k3)
        manual_add_k_val = manual_k(testvals, testvals).evaluate()
        # np.testing.assert_allclose(add_k_val.detach().numpy(), manual_add_k_val.detach().numpy(), atol=1e-5)
        self.assertTrue(torch.allclose(add_k_val, manual_add_k_val, atol=1e-5))

    def test_optimizing(self):
        # This tests should pass so long as nothing breaks.
        torch.random.manual_seed(1)
        data = torch.randn(40, 4)
        target = torch.sin(data).sum(dim=-1)
        d = 4

        AddK = NewtonGirardAdditiveKernel(RBFKernel(ard_num_dims=d), d, max_degree=3)

        class TestGPModel(ExactGP):
            def __init__(self, train_x, train_y, likelihood, kernel):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean()
                self.covar_module = kernel

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)

        model = TestGPModel(data, target, GaussianLikelihood(), ScaleKernel(AddK))
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.train()
        for i in range(2):
            optim.zero_grad()
            out = model(data)
            loss = -mll(out, target)
            loss.backward()
            optim.step()

    def test_ard(self):
        base_k = RBFKernel(ard_num_dims=3)
        base_k.initialize(lengthscale=[1.0, 2.0, 3.0])
        AddK = NewtonGirardAdditiveKernel(base_k, 3, max_degree=1)

        testvals = torch.tensor([[1, 2, 3], [7, 5, 2]], dtype=torch.float)
        add_k_val = AddK(testvals, testvals).evaluate()

        ks = []
        for i in range(3):
            k = RBFKernel(active_dims=i)
            k.initialize(lengthscale=i + 1)
            ks.append(k)
        manual_k = ScaleKernel(AdditiveKernel(*ks))
        manual_k.initialize(outputscale=1.0)
        manual_add_k_val = manual_k(testvals, testvals).evaluate()

        # np.testing.assert_allclose(add_k_val.detach().numpy(), manual_add_k_val.detach().numpy(), atol=1e-5)
        self.assertTrue(torch.allclose(add_k_val, manual_add_k_val, atol=1e-5))

    def test_diag(self):
        AddK = NewtonGirardAdditiveKernel(RBFKernel(ard_num_dims=3), 3, 2)
        self.assertEqual(AddK.base_kernel.lengthscale.numel(), 3)
        self.assertEqual(AddK.outputscale.numel(), 2)

        testvals = torch.tensor([[1, 2, 3], [7, 5, 2]], dtype=torch.float)
        add_k_val = AddK(testvals, testvals).diag()

        manual_k1 = ScaleKernel(
            AdditiveKernel(RBFKernel(active_dims=0), RBFKernel(active_dims=1), RBFKernel(active_dims=2))
        )
        manual_k1.initialize(outputscale=1 / 2)
        manual_k2 = ScaleKernel(
            AdditiveKernel(RBFKernel(active_dims=[0, 1]), RBFKernel(active_dims=[1, 2]), RBFKernel(active_dims=[0, 2]))
        )
        manual_k2.initialize(outputscale=1 / 2)
        manual_k = AdditiveKernel(manual_k1, manual_k2)
        manual_add_k_val = manual_k(testvals, testvals).diag()

        # np.testing.assert_allclose(add_k_val.detach().numpy(), manual_add_k_val.detach().numpy(), atol=1e-5)
        self.assertTrue(torch.allclose(add_k_val, manual_add_k_val, atol=1e-5))
