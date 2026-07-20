#!/usr/bin/env python3


import math
import torch
from torch import distributions
from torch.distributions import Distribution as TDistribution
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal, lazy_property
from torch.distributions.multivariate_normal import _batch_mv, _batch_mahalanobis

class MultivariateStudentT(TDistribution):
    r"""
    Creates a multivariate student t distribution
    parameterized by degrees of freedoms, a mean vector and a covariance matrix.
    Args:
        df (float or Tensor): degrees of freedom
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """

    arg_constraints = {
                    'df': constraints.positive,
                    'loc': constraints.real_vector,
                    'scale': constraints.positive_definite}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, df, loc, scale, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")

        if df <= 2:
            raise ValueError("df must be greater than 2")


        if scale.dim() < 2:
            raise ValueError("scale must be at least two-dimensional, "
                                "with optional leading batch dimensions")
        batch_shape = torch.broadcast_shapes(scale.shape[:-2], loc.shape[:-1])
        self.scale = scale.expand(batch_shape + (-1, -1))
        self.loc = loc.expand(batch_shape + (-1,))
        self.df= df

        event_shape = self.loc.shape[-1:]
        super(MultivariateStudentT, self).__init__(batch_shape, event_shape, validate_args=validate_args)

        self._unbroadcasted_scale_tril = torch.linalg.cholesky(scale)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateStudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        scale_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if 'scale' in self.__dict__:
            new.scale = self.scale.expand(scale_shape)
        if 'scale_tril' in self.__dict__:
            new.scale_tril = self.scale_tril.expand(scale_shape)
        if 'precision_matrix' in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(scale_shape)
        super(MultivariateStudentT, new).__init__(batch_shape,
                                                self.event_shape,
                                                validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def covariance_matrix(self):
        return (self.df / (self.df - 2)) * (torch.matmul(self._unbroadcasted_scale_tril,
                             self._unbroadcasted_scale_tril.mT)
                .expand(self._batch_shape + self._event_shape + self._event_shape))

    @lazy_property
    def precision_matrix(self):
        return((self.df - 2)/self.df) *  torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (self.df / (self.df - 2)) * self._unbroadcasted_scale_tril.pow(2).sum(-1).expand(
            self._batch_shape + self._event_shape)
    
    """
    Y \sim N(0, \Sigma)
    U \sim \Chi^2_v
    X \sim t_v(\mu, \Sigma)
    X = \mu + Y \frac{v}{U}
    """
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        Y = _batch_mv(self._unbroadcasted_scale_tril, eps)
        chi_v = distributions.Chi2(self.df)
        U = chi_v.sample(shape)
        return self.loc + Y * torch.sqrt(self.df / U)


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        n = self._event_shape[0]
        diff = (value - self.loc).unsqueeze(-1)
        res = (-(self.df + n)/2) * torch.log(1 + (1/self.df) * (torch.transpose(diff, -2, -1) @ torch.linalg.inv(self.scale) @ diff))
        res += -0.5 * torch.logdet(self.scale)
        res += - (n/2) * torch.log(torch.tensor(torch.pi))
        res += - (n/2) * torch.log(torch.tensor(self.df))
        res += - torch.lgamma(torch.tensor(self.df/2))
        res += torch.lgamma((self.df + n)/2)
        return torch.squeeze(torch.squeeze(res,-1), -1)


    """
    X \sim t_v(\mu, \Sigma)
    H(X) = \frac{1}{2} \logdet{\Sigma} + \log{\frac{(v \pi)^{n/2}}{\Gamma(n/2)} B(n/2, v/2)} + \frac{v+n}{2}[\Digamma(\frac{v+n}{2} - \Digamma{v}{2})]
    """
    def entropy(self):
        n = self._event_shape[0]
        a = 0.5 * torch.logdet(self.scale)
        b = (n/2) * torch.log(torch.pi * self.df) + torch.lgamma(self.df/2) - torch.lgamma((self.df + n)/2)
        c = ((self.df + n)/2) * (torch.digamma((self.df+n)/2) - torch.digamma(self.df/2) )
        return a + b + c
