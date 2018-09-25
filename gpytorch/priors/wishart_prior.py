from __future__ import absolute_import, division, print_function, unicode_literals

from gpytorch.priors.prior import Prior


class WishartPrior(Prior):
    """Wishart prior over n x n positive definite matrices

    pdf(Sigma) ~ |Sigma|^(nu - n - 1)/2 * exp(-0.5 * Trace(K^-1 Sigma))

    where nu > n - 1 are the degrees of freedom and K > 0 is the p x p scale matrix
    """

    def __init__(self, nu, K):
        # TODO: Derive from torch Distribution
        raise NotImplementedError()


class InverseWishartPrior(Prior):
    """Inverse Wishart prior over n x n positive definite matrices

    pdf(Sigma) ~ |Sigma|^-(nu + 2 * n)/2 * exp(-0.5 * Trace(K Sigma^-1))

    where nu > 0 are the degrees of freedom and K > 0 is the p x p scale matrix
    """

    def __init__(self, nu, K):
        # TODO: Derive from torch Distribution
        raise NotImplementedError()
