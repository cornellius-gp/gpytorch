#!/usr/bin/env python3

import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.nn import Module as TModule

from .prior import Prior


class WishartPrior(Prior):
    """Wishart prior over n x n positive definite matrices

    pdf(Sigma) ~ |Sigma|^(nu - n - 1)/2 * exp(-0.5 * Trace(K^-1 Sigma))

    where nu > n - 1 are the degrees of freedom and K > 0 is the p x p scale matrix

    Reference: A. Shah, A. G. Wilson, and Z. Ghahramani. Student-t Processes as
        Alternatives to Gaussian Processes. ArXiv e-prints, Feb. 2014.
    """

    arg_constraints = {"K_inv": constraints.positive_definite, "nu": constraints.positive}
    support = constraints.positive_definite
    _validate_args = True

    def __init__(self, nu, K, validate_args=False):
        TModule.__init__(self)
        if K.dim() < 2:
            raise ValueError("K must be at least 2-dimensional")
        n = K.shape[-1]
        if K.shape[-2] != K.shape[-1]:
            raise ValueError("K must be square")
        if isinstance(nu, Number):
            nu = torch.tensor(float(nu))
        if torch.any(nu <= n):
            raise ValueError("Must have nu > n - 1")
        self.n = torch.tensor(n, dtype=torch.long, device=nu.device)
        batch_shape = nu.shape
        event_shape = torch.Size([n, n])
        # normalization constant
        logdetK = torch.logdet(K)
        C = -(nu / 2) * (logdetK + n * math.log(2)) - torch.mvlgamma(nu / 2, n)
        K_inv = torch.inverse(K)
        # need to assign values before registering as buffers to make argument validation work
        self.nu = nu
        self.K_inv = K_inv
        self.C = C
        super(WishartPrior, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        # now need to delete to be able to register buffer
        del self.nu, self.K_inv, self.C
        self.register_buffer("nu", nu)
        self.register_buffer("K_inv", K_inv)
        self.register_buffer("C", C)

    def log_prob(self, X):
        # I'm sure this could be done more elegantly
        logdetp = torch.logdet(X)
        Kinvp = torch.matmul(self.K_inv, X)
        trKinvp = torch.diagonal(Kinvp, dim1=-2, dim2=-1).sum(-1)
        return self.C + 0.5 * (self.nu - self.n - 1) * logdetp - trKinvp


class InverseWishartPrior(Prior):
    """Inverse Wishart prior over n x n positive definite matrices

    pdf(Sigma) ~ |Sigma|^-(nu + 2 * n)/2 * exp(-0.5 * Trace(K Sigma^-1))

    where nu > 0 are the degrees of freedom and K > 0 is the p x p scale matrix

    Reference: A. Shah, A. G. Wilson, and Z. Ghahramani. Student-t Processes as
        Alternatives to Gaussian Processes. ArXiv e-prints, Feb. 2014.
    """

    arg_constraints = {"K": constraints.positive_definite, "nu": constraints.positive}
    support = constraints.positive_definite
    _validate_args = True

    def __init__(self, nu, K, validate_args=False):
        TModule.__init__(self)
        if K.dim() < 2:
            raise ValueError("K must be at least 2-dimensional")
        n = K.shape[-1]
        if isinstance(nu, Number):
            nu = torch.tensor(float(nu))
        if torch.any(nu <= 0):
            raise ValueError("Must have nu > 0")
        self.n = torch.tensor(n, dtype=torch.long, device=nu.device)
        batch_shape = nu.shape
        event_shape = torch.Size([n, n])
        # normalization constant
        c = (nu + n - 1) / 2
        logdetK = torch.logdet(K)
        C = c * (logdetK - n * math.log(2)) - torch.mvlgamma(c, n)
        # need to assign values before registering as buffers to make argument validation work
        self.nu = nu
        self.K = K
        self.C = C
        super(InverseWishartPrior, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        # now need to delete to be able to register buffer
        del self.nu, self.K, self.C
        self.register_buffer("nu", nu)
        self.register_buffer("K", K)
        self.register_buffer("C", C)

    def log_prob(self, X):
        logdetp = torch.logdet(X)
        pinvK = torch.solve(self.K, X)[0]
        trpinvK = torch.diagonal(pinvK, dim1=-2, dim2=-1).sum(-1)  # trace in batch mode
        return self.C - 0.5 * ((self.nu + 2 * self.n) * logdetp + trpinvK)
