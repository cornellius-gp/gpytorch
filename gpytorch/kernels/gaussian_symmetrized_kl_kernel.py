#!/usr/bin/env python3

# from torch.distributions import Normal, kl_divergence
from .distributional_input_kernel import DistributionalInputKernel


def _symmetrized_kl(dist1, dist2, eps=1e-8):
    """
    Symmetrized KL distance between two Gaussian distributions. We assume that
    the first half of the distribution tensors are the mean, and the second half
    are the log variances.
    Args:
        :attr:`dist1` (torch.Tensor) has shapes batch x n x dimensions. The first half
            of the last dimensions are the means, while the second half are the log-variances.
        :attr:`dist2` (torch.Tensor) has shapes batch x n x dimensions. The first half
            of the last dimensions are the means, while the second half are the log-variances.
        :attr:`eps` (float) jitter term for the noise variance
    """

    num_dims = int(dist1.shape[-1] / 2)

    dist1_mean = dist1[..., :num_dims].unsqueeze(-3)
    dist1_logvar = dist1[..., num_dims:].unsqueeze(-3)
    dist1_var = eps + dist1_logvar.exp()

    dist2_mean = dist2[..., :num_dims].unsqueeze(-2)
    dist2_logvar = dist2[..., num_dims:].unsqueeze(-2)
    dist2_var = eps + dist2_logvar.exp()

    var_ratio12 = dist1_var / dist2_var
    # log_var_ratio12 = var_ratio12.log()
    # note that the log variance ratio cancels because of the summed KL.
    loc_sqdiffs = (dist1_mean - dist2_mean).pow(2)
    kl1 = 0.5 * (var_ratio12 + loc_sqdiffs / dist2_var - 1)
    kl2 = 0.5 * (var_ratio12.reciprocal() + loc_sqdiffs / dist1_var - 1)
    symmetrized_kl = kl1 + kl2
    return symmetrized_kl.sum(-1).transpose(-1, -2)


class GaussianSymmetrizedKLKernel(DistributionalInputKernel):
    r"""
    Computes a kernel based on the symmetrized KL divergence, assuming that two Gaussian
    distributions are inputted. Inputs are assumed to be `batch x N x 2d` tensors where `d` is the
    dimension of the distribution. The first `d` dimensions are the mean parameters of the
    `batch x N` distributions, while the second `d` dimensions are the log variances.

    Original citation is Moreno et al, '04
    (https://papers.nips.cc/paper/2351-a-kullback-leibler-divergence-based-kernel-for-svm-\
    classification-in-multimedia-applications.pdf) for the symmetrized KL divergence kernel between
    two Gaussian distributions.
    """

    def __init__(self, **kwargs):
        distance_function = _symmetrized_kl
        super(GaussianSymmetrizedKLKernel, self).__init__(distance_function=distance_function, **kwargs)
