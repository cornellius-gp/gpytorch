#!/usr/bin/env python3

import math
import numpy as np
import torch
from torch.nn import Module
from .. import settings


class GaussHermiteQuadrature1D(Module):
    """
    Implements Gauss-Hermite quadrature for integrating a function with respect to several 1D Gaussian distributions
    in batch mode. Within GPyTorch, this is useful primarily for computing expected log likelihoods for variational
    inference.

    This is implemented as a Module because Gauss-Hermite quadrature has a set of locations and weights that it
    should initialize one time, but that should obey parent calls to .cuda(), .double() etc.
    """
    def __init__(self, num_locs=settings.num_gauss_hermite_locs.value()):
        super().__init__()
        self .num_locs = num_locs

        locations, weights = self._locs_and_weights(num_locs)

        self.locations = locations
        self.weights = weights

    def _apply(self, fn):
        self.locations = fn(self.locations)
        self.weights = fn(self.weights)
        return super(GaussHermiteQuadrature1D, self)._apply(fn)

    def _locs_and_weights(self, num_locs):
        """
        Get locations and weights for Gauss-Hermite quadrature. Note that this is **not** intended to be used
        externally, because it directly creates tensors with no knowledge of a device or dtype to cast to.

        Instead, create a GaussHermiteQuadrature1D object and get the locations and weights from buffers.
        """
        locations, weights = np.polynomial.hermite.hermgauss(num_locs)
        locations = torch.Tensor(locations)
        weights = torch.Tensor(weights)
        return locations, weights

    def forward(self, func, gaussian_dists):
        """
        Runs Gauss-Hermite quadrature on the callable func, integrating against the Gaussian distributions specified
        by gaussian_dists.

        Args:
            - func (callable): Function to integrate
            - gaussian_dists (Distribution): Either a MultivariateNormal whose covariance is assumed to be diagonal
                or a :obj:`torch.distributions.Normal`.
        Returns:
            - Result of integrating func against each univariate Gaussian in gaussian_dists.
        """
        mus = gaussian_dists.mean
        vars = gaussian_dists.variance

        shifted_locs = torch.sqrt(2.0 * vars).unsqueeze(-1) * self.locations + mus.unsqueeze(-1)
        res = (1 / math.sqrt(math.pi)) * (func(shifted_locs) * self.weights).sum(-1)

        return res
