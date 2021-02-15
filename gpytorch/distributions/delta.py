#!/usr/bin/env python3

import numbers

import torch
from torch.distributions import constraints
from torch.distributions.kl import register_kl

from .distribution import Distribution
from .multivariate_normal import MultivariateNormal

try:
    from pyro.distributions import Delta

except ImportError:
    # Mostly copied from https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/delta.py
    class Delta(Distribution):
        """
        Degenerate discrete distribution (a single point).

        Discrete distribution that assigns probability one to the single element in
        its support. Delta distribution parameterized by a random choice should not
        be used with MCMC based inference, as doing so produces incorrect results.

        :param torch.Tensor v: The single support element.
        :param torch.Tensor log_density: An optional density for this Delta. This
            is useful to keep the class of :class:`Delta` distributions closed
            under differentiable transformation.
        :param int event_dim: Optional event dimension, defaults to zero.
        """

        arg_constraints = {"v": constraints.real, "log_density": constraints.real}
        has_rsample = True

        def __init__(self, v, log_density=0.0, event_dim=0, validate_args=None):
            if event_dim > v.dim():
                raise ValueError("Expected event_dim <= v.dim(), actual {} vs {}".format(event_dim, v.dim()))
            batch_dim = v.dim() - event_dim
            batch_shape = v.shape[:batch_dim]
            event_shape = v.shape[batch_dim:]
            if isinstance(log_density, numbers.Number):
                log_density = torch.full(batch_shape, log_density, dtype=v.dtype, device=v.device)
            elif validate_args and log_density.shape != batch_shape:
                raise ValueError("Expected log_density.shape = {}, actual {}".format(log_density.shape, batch_shape))
            self.v = v
            self.log_density = log_density
            super().__init__(batch_shape, event_shape, validate_args=validate_args)

        def expand(self, batch_shape, _instance=None):
            new = self._get_checked_instance(Delta, _instance)
            batch_shape = torch.Size(batch_shape)
            new.v = self.v.expand(batch_shape + self.event_shape)
            new.log_density = self.log_density.expand(batch_shape)
            super().__init__(batch_shape, self.event_shape, validate_args=False)
            new._validate_args = self._validate_args
            return new

        def rsample(self, sample_shape=torch.Size()):
            shape = sample_shape + self.v.shape
            return self.v.expand(shape)

        def log_prob(self, x):
            v = self.v.expand(self.batch_shape + self.event_shape)
            log_prob = (x == v).type(x.dtype).log()
            if len(self.event_shape):
                log_prob = log_prob.sum(list(range(-1, -len(self.event_shape) - 1, -1)))
            return log_prob + self.log_density

        @property
        def mean(self):
            return self.v

        @property
        def variance(self):
            return torch.zeros_like(self.v)


@register_kl(Delta, MultivariateNormal)
def kl_mvn_mvn(p_dist, q_dist):
    return -q_dist.log_prob(p_dist.mean)
