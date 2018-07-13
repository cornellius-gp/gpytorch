from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
from torch.distributions.constraints import positive_definite
from gpytorch.priors.prior import Prior


class WishartPrior(Prior):
    """TODO

    """

    def __init__(self, nu, K):
        super(WishartPrior, self).__init__()
        n = K.shape[0]
        if not positive_definite.check(K):
            raise ValueError("K must be positive definite")
        self.register_buffer("K_inv", torch.inverse(K))
        if nu <= n:
            raise ValueError("Must have nu > n - 1")
        self.register_buffer("nu", torch.Tensor([nu]))
        # normalization constant
        C = -(
            nu / 2 * torch.log(torch.det(K))
            + nu * n / 2 * math.log(2)
            + log_mv_gamma(n, nu / 2)
        )
        self.register_buffer("C", C)
        self._log_transform = False

    def _log_prob(self, parameter):
        if not positive_definite.check(parameter):
            raise ValueError("parameter must be positive definite for Wishart prior")
        return self.C + 0.5 * (
            (self.nu - self.shape[0] - 1) * torch.log(torch.det(parameter))
            - torch.trace(self.K_inv.matmul(parameter))
        )

    def is_in_support(self, parameter):
        return bool(positive_definite.check(parameter))

    @property
    def shape(self):
        return self.K_inv.shape


class InverseWishartPrior(Prior):
    """TODO

    """

    def __init__(self, nu, K):
        super(InverseWishartPrior, self).__init__()
        n = K.shape[0]
        if not positive_definite.check(K):
            raise ValueError("K must be positive definite")
        self.register_buffer("K", K)
        if nu <= 0:
            raise ValueError("Must have nu > 0")
        self.register_buffer("nu", torch.Tensor([nu]))
        # normalization constant
        c = (nu + n - 1) / 2
        C = c * torch.log(torch.det(K)) - c * n * math.log(2) - log_mv_gamma(n, c)
        self.register_buffer("C", C)
        self._log_transform = False

    def _log_prob(self, parameter):
        if not positive_definite.check(parameter):
            raise ValueError("parameter must be positive definite for Inverse Wishart prior")
        return self.C - 0.5 * (
            (self.nu + 2 * self.shape[0]) * torch.log(torch.det(parameter))
            + torch.trace(self.K.matmul(torch.inverse(parameter)))
        )

    def is_in_support(self, parameter):
        return bool(positive_definite.check(parameter))

    @property
    def shape(self):
        return self.K.shape


def log_mv_gamma(p, a):
    """Simple implementation of the log multivariate gamma function Gamma_p(a)

    Args:
        p (int): dimension
        a (float): argument

    Will be made obsolete by https://github.com/pytorch/pytorch/issues/9378
    """
    C = p * (p - 1) / 4 * math.log(math.pi)
    return float(C + torch.lgamma(a - 0.5 * torch.arange(p, dtype=torch.float)).sum())
