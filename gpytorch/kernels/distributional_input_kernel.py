#!/usr/bin/env python3

from torch.distributions import Normal, kl_divergence

from .kernel import Kernel


def _symmetrized_kl(dist1, dist2, num_dims=2, eps=1e-8):
    dist1_mean = dist1[..., :num_dims].unsqueeze(-3)
    dist1_logvar = dist1[..., num_dims:].unsqueeze(-2)
    dist1_var = eps + dist1_logvar.exp()

    dist2_mean = dist2[..., :num_dims].unsqueeze(-3)
    dist2_logvar = dist2[..., num_dims:].unsqueeze(-2)
    dist2_var = eps + dist2_logvar.exp()

    dist1 = Normal(dist1_mean, dist1_var ** 0.5)
    dist2 = Normal(dist2_mean, dist2_var ** 0.5)

    res = (kl_divergence(dist1, dist2) + kl_divergence(dist2, dist1)).sum(-1)
    return res.transpose(-1, -2)


class DistributionalInputKernel(Kernel):
    r"""
    Computes a covariance matrix over __Gaussian__ distributions via exponentiating the
    distance function between probability distributions.
    .. math::

        \begin{equation*}
            k(p(x), p(x')) = \exp\{-a d(p(x), p(x'))\})
        \end{equation*}

    where :math:`a` is the lengthscale.
    The implemented distance functions comes from Moreno et al, '04
    (https://papers.nips.cc/paper/2351-a-kullback-leibler-divergence-based-kernel-for-svm-\
    classification-in-multimedia-applications.pdf) for the symmetrized KL. For
    general distributions, these kernels may not produce PSD kernels; however, they do for
    Gaussian distributions which are implemented here.

    Args:
        :attr:`distance` (string, optional, default=`symmetrized_kl`):
            `symmetrized_kl`. Determines whether to use the
            exponentiated symmetrized KL divergence.
        :attr:`num_dims` (int): The first `num_dims` dimensions of the last dimension are
            assumed to be the mean. The second num_dims are assumed to be the log variance.
    """
    has_lengthscale = True

    def __init__(self, distance="symmetrized_kl", num_dims=4, **kwargs):
        super(DistributionalInputKernel, self).__init__(**kwargs)
        self.num_dims = num_dims

        if distance == "symmetrized_kl":
            self.distance_function = _symmetrized_kl
        else:
            raise NotImplementedError("Only symmetrized KL has been implemented.")

    def forward(self, x1, x2, diag=False, *args, **kwargs):
        negative_covar_func = -self.distance_function(x1, x2, num_dims=self.num_dims)
        res = negative_covar_func.div(self.lengthscale).exp()

        if not diag:
            return res
        else:
            return res.diag()
