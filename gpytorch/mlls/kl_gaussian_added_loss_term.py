#!/usr/bin/env python3

from torch.distributions import kl_divergence

from ..distributions import MultivariateNormal
from .added_loss_term import AddedLossTerm


class KLGaussianAddedLossTerm(AddedLossTerm):
    r"""
    This class is used by variational GPLVM models.
    It adds the KL divergence between two multivariate Gaussian distributions:
    scaled by the size of the data and the number of output dimensions.

    .. math::

        D_\text{KL} \left( q(\mathbf x) \Vert p(\mathbf x) \right)


    :param q_x: The MVN distribution :math:`q(\mathbf x)`.
    :param p_x: The MVN distribution :math:`p(\mathbf x)`.
    :param n: Size of the latent space.
    :param data_dim: Dimensionality of the :math:`\mathbf Y` values.
    """

    def __init__(self, q_x: MultivariateNormal, p_x: MultivariateNormal, n: int, data_dim: int):
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
