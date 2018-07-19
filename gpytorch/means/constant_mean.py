from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from gpytorch.means import Mean
from gpytorch.priors._compatibility import _bounds_to_prior


class ConstantMean(Mean):
    def __init__(self, prior=None, batch_size=None, constant_bounds=None):
        # TODO: Remove deprecated bounds kwarg
        prior = _bounds_to_prior(prior=prior, bounds=constant_bounds, batch_size=batch_size, log_transform=False)
        super(ConstantMean, self).__init__()
        self.batch_size = batch_size
        self.register_parameter(
            name="constant", parameter=torch.nn.Parameter(torch.zeros(batch_size or 1, 1)), prior=prior
        )

    def forward(self, input):
        return self.constant.expand(input.size(0), input.size(1))
