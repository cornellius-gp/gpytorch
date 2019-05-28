#!/usr/bin/env python3

import math

import torch
from torch.distributions.kl import register_kl
from torch.distributions.utils import lazy_property

import math
import torch
import torch.distributions as dist

from . import MultivariateNormal

from ..lazy import LazyTensor, CholLazyTensor
from .distribution import Distribution
from ..utils.broadcasting import _mul_broadcast_shape



def _batch_mv(bmat, bvec):
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


class MartinDistribution(Distribution):
    def __init__(self, mean, lazy_covariance_matrix, nu):
        assert isinstance(lazy_covariance_matrix, CholLazyTensor)  # Needs to be a CholLazyTensor for efficiency.
        self.loc = mean
        self.dim = self.loc.size(-1)
        self.lazy_covariance_matrix = lazy_covariance_matrix
        self.nu = nu
        self.half_nu = 0.5 * nu
        self.gamma_dist = dist.Gamma(self.half_nu, self.half_nu)

        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(MartinDistribution, self).__init__(batch_shape, event_shape, validate_args=False)

    #def rsample(self, sample_shape=()):
    #    tau = self.gamma_dist.rsample(sample_shape=sample_shape)
    #    z = torch.randn(sample_shape + (self.dim,)) / tau.sqrt()
    #    return _batch_mv(self.L, z) + self.mu

    def sample(self, sample_shape=torch.Size(), base_samples=None):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)


@register_kl(MartinDistribution, MultivariateNormal)
def kl_mvn_mvn(p_dist, q_dist):
    output_shape = _mul_broadcast_shape(p_dist.batch_shape, q_dist.batch_shape)
    if output_shape != p_dist.batch_shape:
        p_dist = p_dist.expand(output_shape)
    if output_shape != q_dist.batch_shape:
        q_dist = q_dist.expand(output_shape)

    q_mean = q_dist.loc
    q_covar = q_dist.lazy_covariance_matrix

    p_mean = p_dist.loc
    p_covar = p_dist.lazy_covariance_matrix
    root_p_covar = p_covar.root_decomposition().root.evaluate()

    mean_diffs = p_mean - q_mean
    if isinstance(root_p_covar, LazyTensor):
        # right now this just catches if root_p_covar is a DiagLazyTensor,
        # but we may want to be smarter about this in the future
        root_p_covar = root_p_covar.evaluate()

    nu_diag = (p_dist.nu / (p_dist.nu - 2.0)).sqrt().unsqueeze(-2)
    root_p_covar = root_p_covar * nu_diag

    inv_quad_rhs = torch.cat([mean_diffs.unsqueeze(-1), root_p_covar], -1)
    logdet_p_covar = p_covar.logdet()
    trace_plus_inv_quad_form, logdet_q_covar = q_covar.inv_quad_logdet(inv_quad_rhs=inv_quad_rhs, logdet=True)

    # Compute additional entropy terms
    half_nu_one, half_nu = p_dist.half_nu + 0.5, p_dist.half_nu
    entropy1 = half_nu_one * (torch.digamma(half_nu_one) - torch.digamma(half_nu))
    entropy2 = torch.lgamma(half_nu) - torch.lgamma(half_nu_one)
    entropy3 = 0.5 * torch.log(math.pi * p_dist.nu)
    entropy_sum = (entropy1 + entropy2 + entropy3).sum()

    # Compute the KL Divergence.
    res = 0.5 * sum(
        [
            logdet_q_covar,
            logdet_p_covar.mul(-1),
            trace_plus_inv_quad_form,  # Tr(\Sigma_{q}^{-1}L(nu/nu-2)L') + (mu_p - mu_q)\Sigma_{q}^{-1}(mu_p - mu_q)
            math.log(2.0 * math.pi) * float(mean_diffs.size(-1))
        ]
    )
    return res - entropy_sum
