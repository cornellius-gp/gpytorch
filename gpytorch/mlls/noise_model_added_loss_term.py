#!/usr/bin/env python3

from .added_loss_term import AddedLossTerm


class NoiseModelAddedLossTerm(AddedLossTerm):
    def __init__(self, noise_model):
        from .exact_marginal_log_likelihood import ExactMarginalLogLikelihood

        self.noise_mll = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)

    def loss(self, *params):
        output = self.noise_mll.model(*params)
        targets = self.noise_mll.model.train_targets
        return self.noise_mll(output, targets)
