#!/usr/bin/env python3

from .added_loss_term import AddedLossTerm


class InducingPointKernelAddedLossTerm(AddedLossTerm):
    def __init__(self, variational_dist, prior_dist, likelihood):
        self.prior_dist = prior_dist
        self.variational_dist = variational_dist
        self.likelihood = likelihood

    def loss(self, *params):
        prior_covar = self.prior_dist.lazy_covariance_matrix
        variational_covar = self.variational_dist.lazy_covariance_matrix
        diag = prior_covar.diag() - variational_covar.diag()
        shape = prior_covar.shape[:-1]
        noise_diag = self.likelihood._shaped_noise_covar(shape, *params).diag()
        return 0.5 * (diag / noise_diag).sum()
