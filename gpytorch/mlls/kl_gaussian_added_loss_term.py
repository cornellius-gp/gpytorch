#!/usr/bin/env python3

from torch.distributions import kl_divergence

from .added_loss_term import AddedLossTerm


class KLGaussianAddedLossTerm(AddedLossTerm):
    def __init__(self, q_x, p_x, n, data_dim):
        super().__init__()
        self.q_x = q_x
        self.p_x = p_x
        self.n = n
        self.data_dim = data_dim

    def loss(self):
        kl_per_latent_dim = kl_divergence(self.q_x, self.p_x).sum(axis=0)  # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum() / self.n  # scalar
        # inside the forward method of variational ELBO,
        # the added loss terms are expanded (using add_) to take the same
        # shape as the log_lik term (has shape data_dim)
        # so they can be added together. Hence, we divide by data_dim to avoid
        # overcounting the kl term
        return kl_per_point / self.data_dim
