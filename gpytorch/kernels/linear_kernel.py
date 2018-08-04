from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import MatmulLazyVariable, RootLazyVariable
from gpytorch.priors._compatibility import _bounds_to_prior


class LinearKernel(Kernel):
    """
    An implementation of the linear kernel :math:`k(x, z) =(x-offset)(z-offset)' + variance`.

    To implement this efficiently, we use a :obj:`gpytorch.lazy.RootLazyVariable` during training and a
    :math:`gpytorch.lazy.MatmulLazyVariable` during test. These lazy variables represent matrices of the form
    :math:`K = XX^{\top}` and :math:`K = XZ^{\top}`. This makes inference efficient because a matrix-vector product
    :math:`Kv` can be computed as :math:`Kv=X(X^{\top}v)`, where the base multiply :math:`Xv` takes only :math:`O(nd)`
    time and space.
    """

    def __init__(
        self,
        num_dimensions,
        variance_prior=None,
        offset_prior=None,
        active_dims=None,
        variance_bounds=None,
        offset_bounds=None,
    ):
        """
        Args:
            num_dimensions (int): Number of data dimensions to expect. This is necessary to create the offset parameter.
            variance_prior (:obj:`gpytorch.priors.Prior`): Prior over the variance parameter (default `None`).
            offset_prior (:obj:`gpytorch.priors.Prior`): Prior over the offset parameter (default `None`).
            active_dims (list): List of data dimensions to operate on. `len(active_dims)` should equal `num_dimensions`.
            variance_bounds (tuple, deprecated): Min and max value for the variance parameter. Deprecated, and now
                                                 creates a :obj:`gpytorch.priors.SmoothedBoxPrior`.
            offset_bounds (tuple, deprecated): Min and max value for the offset parameter. Deprecated, and now creates a
                                                :obj:'gpytorch.priors.SmoothedBoxPrior'.
        """
        super(LinearKernel, self).__init__(active_dims=active_dims)
        variance_prior = _bounds_to_prior(prior=variance_prior, bounds=variance_bounds, log_transform=False)
        self.register_parameter(name="variance", parameter=torch.nn.Parameter(torch.zeros(1)), prior=variance_prior)
        offset_prior = _bounds_to_prior(prior=offset_prior, bounds=offset_bounds, log_transform=False)
        self.register_parameter(
            name="offset", parameter=torch.nn.Parameter(torch.zeros(1, 1, num_dimensions)), prior=offset_prior
        )

    def forward(self, x1, x2):
        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyVariable when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLazyVariable(x1 - self.offset)
        else:
            prod = MatmulLazyVariable(x1 - self.offset, (x2 - self.offset).transpose(2, 1))

        return prod + self.variance.expand(prod.size())
