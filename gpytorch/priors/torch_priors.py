#!/usr/bin/env python3

import torch
from torch.distributions import Gamma, LogNormal, MultivariateNormal, Normal, Uniform
from torch.nn import Module as TModule

from .prior import Prior
from .utils import _bufferize_attributes, _del_attributes

MVN_LAZY_PROPERTIES = ("covariance_matrix", "scale_tril", "precision_matrix")


class NormalPrior(Prior, Normal):
    """
    Normal (Gaussian) Prior

    pdf(x) = (2 * pi * sigma^2)^-0.5 * exp(-(x - mu)^2 / (2 * sigma^2))

    where mu is the mean and sigma^2 is the variance.
    """

    def __init__(self, loc, scale, validate_args=False, transform=None):
        TModule.__init__(self)
        Normal.__init__(self, loc=loc, scale=scale, validate_args=validate_args)
        _bufferize_attributes(self, ("loc", "scale"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return NormalPrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape))


class LogNormalPrior(Prior, LogNormal):
    """
    Log Normal prior.
    """

    def __init__(self, loc, scale, validate_args=None, transform=None):
        TModule.__init__(self)
        LogNormal.__init__(self, loc=loc, scale=scale, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return LogNormalPrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape))


class UniformPrior(Prior, Uniform):
    """
    Uniform prior.
    """

    def __init__(self, a, b, validate_args=None, transform=None):
        TModule.__init__(self)
        Uniform.__init__(self, a, b, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return UniformPrior(self.low.expand(batch_shape), self.high.expand(batch_shape))


class GammaPrior(Prior, Gamma):
    """Gamma Prior parameterized by concentration and rate

    pdf(x) = beta^alpha / Gamma(alpha) * x^(alpha - 1) * exp(-beta * x)

    were alpha > 0 and beta > 0 are the concentration and rate parameters, respectively.
    """

    def __init__(self, concentration, rate, validate_args=False, transform=None):
        TModule.__init__(self)
        Gamma.__init__(self, concentration=concentration, rate=rate, validate_args=validate_args)
        _bufferize_attributes(self, ("concentration", "rate"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return GammaPrior(self.concentration.expand(batch_shape), self.rate.expand(batch_shape))

    def __call__(self, *args, **kwargs):
        return super(Gamma, self).__call__(*args, **kwargs)


class MultivariateNormalPrior(Prior, MultivariateNormal):
    """Multivariate Normal prior

    pdf(x) = det(2 * pi * Sigma)^-0.5 * exp(-0.5 * (x - mu)' Sigma^-1 (x - mu))

    where mu is the mean and Sigma > 0 is the covariance matrix.
    """

    def __init__(
        self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=False, transform=None
    ):
        TModule.__init__(self)
        MultivariateNormal.__init__(
            self,
            loc=loc,
            covariance_matrix=covariance_matrix,
            precision_matrix=precision_matrix,
            scale_tril=scale_tril,
            validate_args=validate_args,
        )
        _bufferize_attributes(self, ("loc", "_unbroadcasted_scale_tril"))
        self._transform = transform

    def cuda(self, device=None):
        """Applies module-level cuda() call and resets all lazy properties"""
        module = self._apply(lambda t: t.cuda(device))
        _del_attributes(module, MVN_LAZY_PROPERTIES)
        return module

    def cpu(self):
        """Applies module-level cpu() call and resets all lazy properties"""
        module = self._apply(lambda t: t.cpu())
        _del_attributes(module, MVN_LAZY_PROPERTIES)
        return module

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        cov_shape = batch_shape + self.event_shape
        new_loc = self.loc.expand(batch_shape)
        new_scale_tril = self.scale_tril.expand(cov_shape)

        return MultivariateNormalPrior(loc=new_loc, scale_tril=new_scale_tril)
