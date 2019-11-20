#!/usr/bin/env python3

import unittest

import torch

from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.models import IndependentModelList

from .test_exact_gp import TestExactGP


class TestModelListGP(unittest.TestCase):
    def create_model(self, fixed_noise=False):
        data = TestExactGP.create_test_data(self)
        likelihood, labels = TestExactGP.create_likelihood_and_labels(self)
        if fixed_noise:
            noise = 0.1 + 0.2 * torch.rand_like(labels)
            likelihood = FixedNoiseGaussianLikelihood(noise)
        return TestExactGP.create_model(self, data, labels, likelihood)

    def test_forward_eval(self):
        models = [self.create_model() for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        model(torch.rand(3))

    def test_forward_eval_fixed_noise(self):
        models = [self.create_model(fixed_noise=True) for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        model(torch.rand(3))

    def test_get_fantasy_model(self):
        models = [self.create_model() for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        model(torch.rand(3), torch.rand(3))
        fant_x = [torch.randn(2), torch.randn(3)]
        fant_y = [torch.randn(2), torch.randn(3)]
        fmodel = model.get_fantasy_model(fant_x, fant_y)
        fmodel(torch.randn(4))

    def test_get_fantasy_model_fixed_noise(self):
        models = [self.create_model(fixed_noise=True) for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        model(torch.rand(3), torch.rand(3))
        fant_x = [torch.randn(2), torch.randn(3)]
        fant_y = [torch.randn(2), torch.randn(3)]
        fant_noise = [0.1 * torch.ones(2), 0.1 * torch.ones(3)]
        fmodel = model.get_fantasy_model(fant_x, fant_y, noise=fant_noise)
        fmodel(torch.randn(4))


if __name__ == "__main__":
    unittest.main()
