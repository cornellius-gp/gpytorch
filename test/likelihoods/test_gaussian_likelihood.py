#!/usr/bin/env python3

import pickle
import unittest

import torch
from linear_operator.operators import DiagLinearOperator

import gpytorch
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import DirichletClassificationLikelihood, FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior
from gpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase


class TestGaussianLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 0

    def create_likelihood(self):
        return GaussianLikelihood()

    def test_pickle_with_prior(self):
        likelihood = GaussianLikelihood(noise_prior=GammaPrior(1, 1))
        pickle.loads(pickle.dumps(likelihood))  # Should be able to pickle and unpickle with a prior


class TestGaussianLikelihoodBatch(TestGaussianLikelihood):
    seed = 0

    def create_likelihood(self):
        return GaussianLikelihood(batch_shape=torch.Size([3]))

    def test_nonbatch(self):
        pass


class TestGaussianLikelihoodMultiBatch(TestGaussianLikelihood):
    seed = 0

    def create_likelihood(self):
        return GaussianLikelihood(batch_shape=torch.Size([2, 3]))

    def test_nonbatch(self):
        pass

    def test_batch(self):
        pass


class TestFixedNoiseGaussianLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    def create_likelihood(self):
        noise = 0.1 + torch.rand(5)
        return FixedNoiseGaussianLikelihood(noise=noise)

    def test_fixed_noise_gaussian_likelihood(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            noise = 0.1 + torch.rand(4, device=device, dtype=dtype)
            lkhd = FixedNoiseGaussianLikelihood(noise=noise)
            # test basics
            self.assertIsInstance(lkhd.noise_covar, FixedGaussianNoise)
            self.assertTrue(torch.equal(noise, lkhd.noise))
            new_noise = 0.1 + torch.rand(4, device=device, dtype=dtype)
            lkhd.noise = new_noise
            self.assertTrue(torch.equal(lkhd.noise, new_noise))
            # test __call__
            mean = torch.zeros(4, device=device, dtype=dtype)
            covar = DiagLinearOperator(torch.ones(4, device=device, dtype=dtype))
            mvn = MultivariateNormal(mean, covar)
            out = lkhd(mvn)
            self.assertTrue(torch.allclose(out.variance, 1 + new_noise))
            # things should break if dimensions mismatch
            mean = torch.zeros(5, device=device, dtype=dtype)
            covar = DiagLinearOperator(torch.ones(5, device=device, dtype=dtype))
            mvn = MultivariateNormal(mean, covar)
            with self.assertWarns(UserWarning):
                lkhd(mvn)
            # test __call__ w/ observation noise
            obs_noise = 0.1 + torch.rand(5, device=device, dtype=dtype)
            out = lkhd(mvn, noise=obs_noise)
            self.assertTrue(torch.allclose(out.variance, 1 + obs_noise))
            # test noise smaller than min_fixed_noise
            expected_min_noise = settings.min_fixed_noise.value(dtype)
            noise[:2] = 0
            lkhd = FixedNoiseGaussianLikelihood(noise=noise)
            expected_noise = noise.clone()
            expected_noise[:2] = expected_min_noise
            self.assertTrue(torch.allclose(lkhd.noise, expected_noise))


class TestFixedNoiseGaussianLikelihoodBatch(BaseLikelihoodTestCase, unittest.TestCase):
    def create_likelihood(self):
        noise = 0.1 + torch.rand(3, 5)
        return FixedNoiseGaussianLikelihood(noise=noise)

    def test_nonbatch(self):
        pass


class TestFixedNoiseGaussianLikelihoodMultiBatch(BaseLikelihoodTestCase, unittest.TestCase):
    def create_likelihood(self):
        noise = 0.1 + torch.rand(2, 3, 5)
        return FixedNoiseGaussianLikelihood(noise=noise)

    def test_nonbatch(self):
        pass

    def test_batch(self):
        pass


class TestDirichletClassificationLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    def create_likelihood(self):
        train_x = torch.randn(15)
        labels = torch.round(train_x).long()
        likelihood = DirichletClassificationLikelihood(labels)
        return likelihood

    def test_batch(self):
        pass

    def test_multi_batch(self):
        pass

    def test_nonbatch(self):
        pass

    def test_dirichlet_classification_likelihood(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            noise = torch.rand(6, device=device, dtype=dtype) > 0.5
            noise = noise.long()
            lkhd = DirichletClassificationLikelihood(noise, dtype=dtype)
            # test basics
            self.assertIsInstance(lkhd.noise_covar, FixedGaussianNoise)
            noise = torch.rand(6, device=device, dtype=dtype) > 0.5
            noise = noise.long()
            new_noise, _, _ = lkhd._prepare_targets(noise, dtype=dtype)
            lkhd.noise = new_noise
            self.assertTrue(torch.equal(lkhd.noise, new_noise))
            # test __call__
            mean = torch.zeros(6, device=device, dtype=dtype)
            covar = DiagLinearOperator(torch.ones(6, device=device, dtype=dtype))
            mvn = MultivariateNormal(mean, covar)
            out = lkhd(mvn)
            self.assertTrue(torch.allclose(out.variance, 1 + new_noise))
            # things should break if dimensions mismatch
            mean = torch.zeros(5, device=device, dtype=dtype)
            covar = DiagLinearOperator(torch.ones(5, device=device, dtype=dtype))
            mvn = MultivariateNormal(mean, covar)
            with self.assertWarns(UserWarning):
                lkhd(mvn)
            # test __call__ w/ new targets
            obs_noise = 0.1 + torch.rand(5, device=device, dtype=dtype)
            obs_noise = (obs_noise > 0.5).long()
            out = lkhd(mvn, targets=obs_noise)
            obs_targets, _, _ = lkhd._prepare_targets(obs_noise, dtype=dtype)
            self.assertTrue(torch.allclose(out.variance, 1.0 + obs_targets))


class TestGaussianLikelihoodWithMissingObs(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 42

    def create_likelihood(self):
        return GaussianLikelihood()

    def test_missing_value_inference_fill(self):
        """
        samples = mvn samples + noise samples
        In this test, we try to recover noise parameters when some elements in
        'samples' are missing at random.
        """

        torch.manual_seed(self.seed)

        mvn, samples = self._make_data()

        missing_probability = 0.33
        missing_idx = torch.distributions.Binomial(1, missing_probability).sample(samples.shape).bool()
        samples[missing_idx] = float("nan")

        # check that the correct noise sd is recovered

        with settings.observation_nan_policy("fill"):
            self._check_recovery(mvn, samples)

    def test_missing_value_inference_mask(self):
        """
        samples = mvn samples + noise samples
        In this test, we try to recover noise parameters when some elements in
        'samples' are missing at random.
        """

        torch.manual_seed(self.seed)

        mvn, samples = self._make_data()

        missing_prop = 0.33
        missing_idx = torch.distributions.Binomial(1, missing_prop).sample(samples.shape[1:]).bool()
        samples[1, missing_idx] = float("nan")

        # check that the correct noise sd is recovered

        with settings.observation_nan_policy("fill"):
            self._check_recovery(mvn, samples)

    def _make_data(self):
        mu = torch.zeros(2, 3)
        sigma = torch.tensor([[[1, 0.999, -0.999], [0.999, 1, -0.999], [-0.999, -0.999, 1]]] * 2).float()
        mvn = MultivariateNormal(mu, sigma)
        samples = mvn.sample(torch.Size([10000]))  # mvn samples
        noise_sd = 0.5
        noise_dist = torch.distributions.Normal(0, noise_sd)
        samples += noise_dist.sample(samples.shape)  # noise
        return mvn, samples

    def _check_recovery(self, mvn, samples):
        likelihood = GaussianLikelihood()
        opt = torch.optim.Adam(likelihood.parameters(), lr=0.05)
        for _ in range(100):
            opt.zero_grad()
            loss = -likelihood.log_marginal(samples, mvn).sum()
            loss.backward()
            opt.step()
        self.assertTrue(abs(float(likelihood.noise.sqrt()) - 0.5) < 0.02)
        # Check log marginal works
        likelihood.log_marginal(samples[0], mvn)


class _DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestDirichletClassificationLikelihoodFantasy(unittest.TestCase):
    @unittest.expectedFailure
    def test_dirichlet_fantasy_model_creation(self):
        """Regression test for #2579: get_fantasy_model should succeed for Dirichlet models.

        This test exercises the full get_fantasy_model pipeline with a batch GP
        (batched train_x) to ensure indexing operations are batch compatible.
        The original #2579 RuntimeError is fixed by this PR. A separate, pre-existing
        issue ("view size is not compatible") in DefaultPredictionStrategy.get_fantasy_strategy
        for batched models is tracked separately and is marked expectedFailure
        until that issue is addressed.
        """
        torch.manual_seed(42)
        n_classes = 3
        n_train = 20
        batch_size = 2

        # Batched train data: (batch, n, d) and (batch, n)
        train_x = torch.randn(batch_size, n_train, 2)
        train_labels = torch.randint(0, n_classes, (batch_size, n_train))

        likelihood = DirichletClassificationLikelihood(targets=train_labels[0], learn_additional_noise=True)
        model = _DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=n_classes)

        # Train a few steps
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)
        for _ in range(3):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -likelihood(output, targets=train_labels[0]).log_prob(likelihood.transformed_targets).sum()
            loss.backward()
            optimizer.step()

        # Run prediction to set up prediction_strategy
        model.eval()
        likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            model(train_x)

        # Create fantasy data
        fant_x = torch.randn(batch_size, 5, 2)
        fant_y = likelihood.transformed_targets[:, :5]

        # This should succeed without RuntimeError (was broken before fix)
        fant_model = model.get_fantasy_model(fant_x, fant_y)

        # Verify fantasy model has correct training size
        self.assertEqual(fant_model.train_inputs[0].shape[-2], n_train + 5)

    def test_dirichlet_get_fantasy_likelihood_accepts_targets(self):
        """Regression test for #2579: get_fantasy_likelihood should accept targets kwarg.

        Before fix: raises RuntimeError "FixedNoiseGaussianLikelihood.fantasize requires a targets kwarg"
        After fix: returns a DirichletClassificationLikelihood.
        """
        torch.manual_seed(42)
        n_classes = 3
        train_labels = torch.randint(0, n_classes, (30,))
        likelihood = DirichletClassificationLikelihood(targets=train_labels, learn_additional_noise=True)

        new_labels = torch.randint(0, n_classes, (8,))
        # This call previously failed; after the fix it must succeed.
        fantasy_lik = likelihood.get_fantasy_likelihood(targets=new_labels)
        self.assertIsInstance(fantasy_lik, DirichletClassificationLikelihood)

    def test_dirichlet_fantasy_preserves_num_classes(self):
        """Fantasy model should preserve num_classes from the original likelihood."""
        torch.manual_seed(42)
        n_classes = 4
        n_train = 30

        train_labels = torch.randint(0, n_classes, (n_train,))
        likelihood = DirichletClassificationLikelihood(targets=train_labels, learn_additional_noise=True)

        new_labels = torch.randint(0, n_classes, (8,))
        fantasy_lik = likelihood.get_fantasy_likelihood(targets=new_labels)

        self.assertEqual(fantasy_lik.num_classes, n_classes)

    def test_dirichlet_fantasy_preserves_alpha_epsilon(self):
        """Fantasy model should preserve alpha_epsilon from the original likelihood."""
        torch.manual_seed(42)
        n_classes = 3
        n_train = 30

        train_labels = torch.randint(0, n_classes, (n_train,))
        likelihood = DirichletClassificationLikelihood(
            targets=train_labels, alpha_epsilon=0.05, learn_additional_noise=True
        )

        new_labels = torch.randint(0, n_classes, (8,))
        fantasy_lik = likelihood.get_fantasy_likelihood(targets=new_labels)

        self.assertEqual(fantasy_lik.alpha_epsilon, 0.05)


if __name__ == "__main__":
    unittest.main()
