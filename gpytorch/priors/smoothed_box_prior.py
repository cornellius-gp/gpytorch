from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from numbers import Number

import torch
from torch.distributions.normal import Normal
from gpytorch.priors.prior import Prior


class SmoothedBoxPrior(Prior):
    """A smoothed approximation of a uniform prior.

    Has full support on the reals and is differentiable everywhere.

        B = {x: a_i <= x_i <= b_i}
        d(x, B) = min_{x' in B} |x - x'|

        pdf(x) ~ exp(- d(x, B)**2 / sqrt(2 * pi * sigma**2))

    """

    def __init__(self, a, b, sigma=0.01, log_transform=False, size=None):
        if isinstance(a, Number) and isinstance(b, Number):
            a = torch.full((size or 1,), float(a))
            b = torch.full((size or 1,), float(b))
        elif not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise ValueError("a and b must be both either scalars or Tensors")
        elif a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        elif size is not None:
            raise ValueError("can only set size for scalar a and b")
        if (b < a).any():
            raise ValueError("must have that a < b (element-wise)")
        super(SmoothedBoxPrior, self).__init__()
        self.register_buffer("a", a.view(-1).clone())
        self.register_buffer("b", b.view(-1).clone())
        self.register_buffer(
            "sigma",
            torch.full_like(self.a, float(sigma))
            if isinstance(sigma, Number)
            else sigma.view(self.a.shape).clone(),
        )
        self.register_buffer("_loc", torch.zeros_like(self.sigma))
        self._initialize_distributions()
        self._log_transform = log_transform

    def _initialize_distributions(self):
        self._tails = [
            Normal(loc=l, scale=s, validate_args=True)
            for l, s in zip(self._loc, self.sigma)
        ]

    @property
    def _c(self):
        return (self.a + self.b) / 2

    @property
    def _r(self):
        return (self.b - self.a) / 2

    @property
    def _M(self):
        # normalization factor to make this a probability distribution
        return torch.log(1 + (self.b - self.a) / (math.sqrt(2 * math.pi) * self.sigma))

    def _log_prob(self, parameter):
        # x = "distances from box`"
        X = ((parameter.view(self.a.shape) - self._c).abs_() - self._r).clamp(min=0)
        return sum(p.log_prob(x) for p, x in zip(self._tails, X)) - self._M.sum()

    def is_in_support(self, parameter):
        return True

    @property
    def shape(self):
        return torch.Size([len(self.tails)])
