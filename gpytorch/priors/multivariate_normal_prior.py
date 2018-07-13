from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from .prior import TorchDistributionPrior


class MultivariateNormalPrior(TorchDistributionPrior):
    def __init__(
        self,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        log_transform=False,
    ):
        if not torch.is_tensor(loc):
            raise ValueError("loc must be a torch Tensor")
        super(MultivariateNormalPrior, self).__init__()
        # let's be lazy and not do the conversion to scale_tril ourselves
        mvn = MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix,
            precision_matrix=precision_matrix,
            scale_tril=scale_tril,
            validate_args=True,
        )
        self.register_buffer("loc", mvn.loc.clone())
        self.register_buffer("scale_tril", mvn.scale_tril.clone())
        self._initialize_distributions()
        self._log_transform = log_transform

    def _initialize_distributions(self):
        self._distribution = MultivariateNormal(
            loc=self.loc, scale_tril=self.scale_tril
        )

    def _log_prob(self, parameter):
        return self._distribution.log_prob(parameter.view(-1))

    def is_in_support(self, parameter):
        return True

    @property
    def shape(self):
        return torch.Size([self.loc.nelement()])
