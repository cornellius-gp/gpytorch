#!/usr/bin/env python3

import math
import pickle
import unittest
import warnings
from collections import OrderedDict

import torch

import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.means import ConstantMean
from gpytorch.priors import NormalPrior
from gpytorch.test.base_mean_test_case import BaseMeanTestCase
from gpytorch.utils.warnings import OldVersionWarning


# Test class for loading models that have state dicts with the old ConstantMean parameter names
class _GPModel(gpytorch.models.ExactGP):
    def __init__(self, mean_module):
        train_x = torch.randn(10, 3)
        train_y = torch.randn(10)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module


class TestConstantMean(BaseMeanTestCase, unittest.TestCase):
    batch_shape = None

    def create_mean(self, prior=None, constraint=None):
        return ConstantMean(
            constant_prior=prior,
            constant_constraint=constraint,
            batch_shape=(self.__class__.batch_shape or torch.Size([])),
        )

    def test_prior(self):
        if self.batch_shape is None:
            prior = NormalPrior(0.0, 1.0)
        else:
            prior = NormalPrior(torch.zeros(self.batch_shape), torch.ones(self.batch_shape))
        mean = self.create_mean(prior=prior)
        self.assertEqual(mean.mean_prior, prior)
        pickle.loads(pickle.dumps(mean))  # Should be able to pickle and unpickle with a prior
        value = prior.sample()
        mean._constant_closure(mean, value)
        self.assertTrue(torch.equal(mean.constant.data, value.reshape(mean.constant.data.shape)))

    def test_constraint(self):
        mean = self.create_mean()
        self.assertAllClose(mean.constant, torch.zeros(mean.constant.shape))

        constraint = GreaterThan(1.5)
        mean = self.create_mean(constraint=constraint)
        self.assertTrue(torch.all(mean.constant >= 1.5))
        mean.constant = torch.full(self.__class__.batch_shape or torch.Size([]), fill_value=1.65)
        self.assertAllClose(mean.constant, torch.tensor(1.65).expand(mean.constant.shape))

    def test_loading_old_module(self):
        batch_shape = self.__class__.batch_shape or torch.Size([])
        constant = torch.randn(batch_shape)
        mean = self.create_mean()
        model = _GPModel(mean)

        old_state_dict = OrderedDict(
            [
                ("likelihood.noise_covar.raw_noise", torch.tensor([0.0])),
                ("likelihood.noise_covar.raw_noise_constraint.lower_bound", torch.tensor(1.0000e-04)),
                ("likelihood.noise_covar.raw_noise_constraint.upper_bound", torch.tensor(math.inf)),
                ("mean_module.constant", constant.unsqueeze(-1)),
            ]
        )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always", OldVersionWarning)
            model.load_state_dict(old_state_dict)
            self.assertTrue(any(issubclass(w.category, OldVersionWarning) for w in ws))
            self.assertEqual(model.mean_module.constant.data, constant)

        new_state_dict = OrderedDict(
            [
                ("likelihood.noise_covar.raw_noise", torch.tensor([0.0])),
                ("likelihood.noise_covar.raw_noise_constraint.lower_bound", torch.tensor(1.0000e-04)),
                ("likelihood.noise_covar.raw_noise_constraint.upper_bound", torch.tensor(math.inf)),
                ("mean_module.raw_constant", constant),
            ]
        )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always", OldVersionWarning)
            model.load_state_dict(new_state_dict)
            self.assertFalse(any(issubclass(w.category, OldVersionWarning) for w in ws))


class TestConstantMeanBatch(TestConstantMean, unittest.TestCase):
    batch_shape = torch.Size([3])


class TestConstantMeanMultiBatch(TestConstantMean, unittest.TestCase):
    batch_shape = torch.Size([2, 3])
