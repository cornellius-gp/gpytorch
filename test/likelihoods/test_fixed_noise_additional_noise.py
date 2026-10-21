#!/usr/bin/env python3

import unittest

import torch
from linear_operator.operators import DiagLinearOperator

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood


class TestFixedAdditionalNoise(unittest.TestCase):
    def test_override_replaces_fixed_noise_and_keeps_learned_noise(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for batch in (torch.Size(), torch.Size([2])):
                for pass_inputs in (False, True):
                    for learned in (False, True):
                        with self.subTest(device=device, batch=batch, inputs=pass_inputs, learned=learned):
                            like = torch.zeros((), device=device, dtype=torch.double)
                            likelihood = FixedNoiseGaussianLikelihood(
                                noise=like.new_full((*batch, 4), 0.1),
                                learn_additional_noise=learned,
                                batch_shape=batch,
                            ).to(like)
                            if learned:
                                likelihood.second_noise = like.new_full((*batch, 1), 0.3)
                            override = like.new_tensor([0.4, 0.5]).expand(*batch, 2).clone()
                            distribution = MultivariateNormal(
                                like.new_zeros(*batch, 2), DiagLinearOperator(like.new_ones(*batch, 2))
                            )
                            inputs = (like.new_zeros(*batch, 2, 1),) if pass_inputs else ()
                            observed = likelihood(distribution, *inputs, noise=override)
                            expected_noise = override + (likelihood.second_noise if learned else 0)
                            torch.testing.assert_close(observed.variance, 1 + expected_noise, atol=1e-14, rtol=1e-14)
                            conditional = likelihood(like.new_zeros(*batch, 2), *inputs, noise=override)
                            torch.testing.assert_close(conditional.variance, expected_noise, atol=1e-14, rtol=1e-14)

    def test_override_and_learned_parameter_gradients_and_mixed_hvp(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                like = torch.zeros((), device=device, dtype=torch.double)
                likelihood = FixedNoiseGaussianLikelihood(
                    noise=like.new_tensor([0.1, 0.2]), learn_additional_noise=True
                ).to(like)
                likelihood.second_noise = like.new_tensor(0.3)
                override = like.new_tensor([0.4, 0.5], requires_grad=True)
                distribution = MultivariateNormal(like.new_zeros(2), DiagLinearOperator(like.new_ones(2)))
                actual = likelihood(distribution, noise=override).variance
                expected = 1 + override + likelihood.second_noise
                leaves = override, likelihood.second_noise_covar.raw_noise

                def derivatives(value):
                    gradient = torch.autograd.grad(value.log().sum(), leaves, create_graph=True)
                    hvp = torch.autograd.grad(sum(g.square().sum() for g in gradient), leaves)
                    return (*gradient, *hvp)

                for actual_gradient, expected_gradient in zip(derivatives(actual), derivatives(expected), strict=True):
                    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-13, rtol=1e-13)


if __name__ == "__main__":
    unittest.main()
