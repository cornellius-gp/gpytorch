#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.module import Module
from gpytorch.priors import GammaPrior, HalfNormalPrior, MultivariateNormalPrior
from gpytorch.test.base_test_case import BaseTestCase


class TestModule(BaseTestCase, unittest.TestCase):
    def test_register_prior(self):
        mock_prior = Module()
        m = Module()
        with self.assertRaises(AttributeError):
            m.register_prior("mock_prior", mock_prior, "foo")
        m.register_parameter("foo", torch.nn.Parameter(torch.zeros(1)))
        with self.assertRaises(RuntimeError):
            m.register_prior("mock_prior", mock_prior, "foo", unittest.mock.Mock())
        m.register_prior("mock_prior", mock_prior, "foo")
        with self.assertRaises(ValueError):
            m.register_prior("mock_prior", mock_prior, lambda: None)
        m.register_prior("mock_prior", mock_prior, lambda m: None)
        with self.assertRaises(ValueError):
            m.register_prior("mock_prior", mock_prior, lambda m: None, lambda v: None)
        m.register_prior("mock_prior", mock_prior, lambda m: None, lambda m, v: None)

    def test_sample_from_prior_ard_batch_and_multivariate(self):
        torch.manual_seed(0)
        kernel = RBFKernel(
            ard_num_dims=4,
            batch_shape=torch.Size([3]),
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
        kernel.sample_from_prior("lengthscale_prior")
        self.assertEqual(kernel.lengthscale.shape, torch.Size([3, 1, 4]))
        self.assertEqual(kernel.lengthscale.unique().numel(), 12)

        kernel_hn = RBFKernel(
            ard_num_dims=4,
            batch_shape=torch.Size([2]),
            lengthscale_prior=HalfNormalPrior(1.0),
        )
        kernel_hn.sample_from_prior("lengthscale_prior")
        self.assertEqual(kernel_hn.lengthscale.shape, torch.Size([2, 1, 4]))
        self.assertEqual(kernel_hn.lengthscale.unique().numel(), 8)

        m = Module()
        m.register_parameter("vec", torch.nn.Parameter(torch.zeros(2, 3)))
        m.register_prior("vec_prior", MultivariateNormalPrior(torch.zeros(3), torch.eye(3)), "vec")
        m.sample_from_prior("vec_prior")
        self.assertEqual(m.vec.shape, torch.Size([2, 3]))
        self.assertFalse(torch.allclose(m.vec[0], m.vec[1]))

    def test_random_module_initialize_nested_paths_and_properties(self):
        parent = Module()
        parent.covar_module = ScaleKernel(RBFKernel(ard_num_dims=2))
        rm = parent.to_random_module()

        ls = torch.tensor([[1.5, 2.5]], requires_grad=True)
        os = torch.tensor(3.25, requires_grad=True)
        rm.initialize(
            **{
                "covar_module.base_kernel.lengthscale": ls,
                "covar_module.outputscale": os,
            }
        )
        self.assertAllClose(rm.covar_module.base_kernel.lengthscale, ls)
        self.assertAllClose(rm.covar_module.outputscale, os)
        self.assertNotIsInstance(rm.covar_module.base_kernel.raw_lengthscale, torch.nn.Parameter)

        raw_ls = torch.tensor([[0.1, 0.2]], requires_grad=True)
        rm.initialize(**{"covar_module.base_kernel.raw_lengthscale": raw_ls})
        self.assertAllClose(rm.covar_module.base_kernel.raw_lengthscale, raw_ls)


if __name__ == "__main__":
    unittest.main()
