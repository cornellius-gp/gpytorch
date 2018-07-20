from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from .prior import TorchDistributionPrior


class MultivariateNormalPrior(TorchDistributionPrior):
    """Multivariate Normal prior

    pdf(x) = det(2 * pi * Sigma)^-0.5 * exp(-0.5 * (x - mu)' Sigma^-1 (x - mu))

    where mu is the mean and Sigma > 0 is the covariance matrix.
    """

    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, log_transform=False):
        if not torch.is_tensor(loc):
            raise ValueError("loc must be a torch Tensor")
        # let's be lazy and not do the conversion to scale_tril ourselves
        mvn = MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix,
            precision_matrix=precision_matrix,
            scale_tril=scale_tril,
            validate_args=True,
        )
        super(MultivariateNormalPrior, self).__init__()
        self.register_buffer("loc", mvn.loc.clone())
        self.register_buffer("scale_tril", mvn.scale_tril.clone())
        self._log_transform = log_transform
        self._initialize_distributions()

    def _initialize_distributions(self):
        self._distribution = MultivariateNormal(loc=self.loc, scale_tril=self.scale_tril)

    def _log_prob(self, parameter):
        return self._distribution.log_prob(parameter.view(-1))

    def is_in_support(self, parameter):
        return True

    def size(self):
        return torch.Size([self.loc.nelement()])
