#!/usr/bin/env python3

import torch

from .kernel import Kernel


class DistributionalInputKernel(Kernel):
    r"""
    Computes a covariance matrix over __Gaussian__ distributions via exponentiating the
    distance function between probability distributions.
    .. math::

        \begin{equation*}
            k(p(x), p(x')) = \exp\{-a d(p(x), p(x'))\})
        \end{equation*}

    where :math:`a` is the lengthscale.

    Args:
        :attr:`distance_function` (function) distance function between distributional inputs.
    """
    has_lengthscale = True

    def __init__(self, distance_function, **kwargs):
        super(DistributionalInputKernel, self).__init__(**kwargs)
        if distance_function is None:
            raise NotImplementedError("DistributionalInputKernel requires a distance function.")

        self.distance_function = distance_function

    def forward(self, x1, x2, diag=False, *args, **kwargs):
        negative_covar_func = -self.distance_function(x1, x2)
        res = negative_covar_func.div(self.lengthscale).exp()

        if not diag:
            return res
        else:
            if torch.is_tensor(res):
                return res.diagonal(dim1=-1, dim2=-2)
            else:
                return res.diag()  # For LazyTensor
