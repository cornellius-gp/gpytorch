#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.likelihoods import HeteroskedasticNoise


class NumericallyUnstableModelExample(gpytorch.models.GP):
    def __init__(self):
        super(NumericallyUnstableModelExample, self).__init__()
        self.fail_arithmetic = False

    def train(self, mode=True):
        if mode:
            self.fail_arithmetic = False  # reset on .train()
        super().train(mode=mode)

    def forward(self, x):
        if self.fail_arithmetic:
            raise ArithmeticError()
        return gpytorch.distributions.MultivariateNormal(torch.tensor([-3.0]), torch.tensor([[2.0]]))


class TestNoiseModels(unittest.TestCase):
    def test_heteroskedasticnoise_error(self):
        noise_model = NumericallyUnstableModelExample().to(torch.double)
        likelihood = HeteroskedasticNoise(noise_model)
        self.assertEqual(noise_model.training, True)
        self.assertEqual(likelihood.training, True)
        noise_model.fail_arithmetic = True
        test_x = torch.tensor([[3.0]])
        with self.assertRaises(ArithmeticError):
            likelihood(test_x)
        self.assertEqual(likelihood.training, True)
        likelihood(test_x)
