from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import torch
from gpytorch.priors import SmoothedBoxPrior

logger = logging.getLogger()


def _bounds_to_prior(
    prior, bounds, batch_size=None, log_transform=True
):
    if prior is not None:
        return prior
    else:
        logger.warning("bounds are deprecated, use a prior instead!")
        a = torch.full((batch_size or 1,), float(bounds[0]))
        b = torch.full((batch_size or 1,), float(bounds[1]))
        return SmoothedBoxPrior(a, b, log_transform=log_transform)
