#!/usr/bin/env python3

import torch
from .kernel import Kernel


def phi_0(r):
    return torch.max(torch.tensor(0.0), 1 - r)  # np.maximum(0, 1-r)


def phi_1(r):
    return phi_0(r) ** 3 * (3 * r + 1)


def phi_2(r):
    return phi_0(r) ** 5 * (8 * r ** 2 + 5 * r + 1)


def wendland_kernel_function(x1, x2, k=0, lambdas=1):
    weighted_element_difference = torch.abs(x1 - x2).div(lambdas)
    if k == 0:
        phi = phi_0(weighted_element_difference)
    elif k == 1:
        phi = phi_1(weighted_element_difference)
    elif k == 2:
        phi = phi_2(weighted_element_difference)
    else:
        raise NotImplementedError()
    return torch.prod(phi)


class WendlandKernel(Kernel):
    """


    """

    def __init__(self, k=0, **kwargs):
        super().__init__(has_lengthscale=True, **kwargs)
        self.k = k

    def forward(self, x1, x2, diag=False, **params):
        dm = torch.empty((x1.shape[0], x2.shape[0]))
        for i in torch.arange(x1.shape[0]):
            for j in torch.arange(x2.shape[0]):
                dm[i, j] = wendland_kernel_function(x1[i], x2[j], self.k, lambdas=self.lengthscale)
        return dm
