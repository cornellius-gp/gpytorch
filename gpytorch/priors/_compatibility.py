from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import torch
from gpytorch.priors import SmoothedBoxPrior


def _bounds_to_prior(prior, bounds, batch_size=None, log_transform=True):
    if prior is not None:
        return prior
    elif bounds is not None:
        warnings.warn(
            "Parameter bounds have been deprecated and will be removed in a future release. "
            "Please either remove them or use a SmoothedBoxPrior instead!", DeprecationWarning
        )
        a = torch.full((batch_size or 1,), float(bounds[0]))
        b = torch.full((batch_size or 1,), float(bounds[1]))
        return SmoothedBoxPrior(a, b, log_transform=log_transform)
    else:
        return None
