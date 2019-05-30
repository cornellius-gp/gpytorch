#!/usr/bin/env python3

import torch
import gpytorch
import unittest
from gpytorch.models import IndependentModelList
from test.models.test_exact_gp import TestExactGP, ExactGPModel


class TestModelListGP(unittest.TestCase):

    def create_model(self):
        data = TestExactGP.create_test_data(self)
        likelihood, labels = TestExactGP.create_likelihood_and_labels(self)
        return TestExactGP.create_model(self, data, labels, likelihood)

    def test_forward_eval(self):
        models = [self.create_model() for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        output = model(torch.rand(3))

    def test_get_fantasy_model(self):
        models = [self.create_model() for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        output = model(torch.rand(3), torch.rand(3))
        fant_x = [torch.randn(2), torch.randn(3)]
        fant_y = [torch.randn(2), torch.randn(3)]
        fmodel = model.get_fantasy_model(fant_x, fant_y)
        fmodel(torch.randn(4))


if __name__ == "__main__":
    unittest.main()
