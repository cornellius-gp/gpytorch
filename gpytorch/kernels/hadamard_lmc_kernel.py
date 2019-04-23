#!/usr/bin/env python3

import torch
from copy import deepcopy
from torch.nn import ModuleList
from ..lazy import lazify
from .kernel import Kernel


class HadamardLMCKernel(Kernel):
    """ Follows Alvarez 2012 """

    # TODO: Plan:
    # i) start without proving the option for the same kernel to be used
    #    for multiple latent functions
    # ii) Add that option
    # iii) How do batches work in GPytorch

    def __init__(self, latent_kernels, num_tasks, num_latent, **kwargs):
        super(HadamardLMCKernel, self).__init__(**kwargs)

        if isinstance(latent_kernels, Kernel):
            latent_kernels = [latent_kernels]

        if not isinstance(latent_kernels, list) or (
                len(latent_kernels) != 1 and
                len(latent_kernels) != num_latent):
            raise RuntimeError(
                "latent_kernels should be a list of means of length either 1 or num_tasks"
            )

        if len(latent_kernels) == 1:
            latent_kernels = latent_kernels + [
                deepcopy(latent_kernels[0]) for i in range(num_latent - 1)
            ]

        # num_pairwise = ((num_tasks * (num_tasks - 1)) // 2) + num_tasks

        self.register_parameter(
            name="kernel_coefficients",
            parameter=torch.nn.Parameter(
                torch.ones(num_latent, num_tasks)))

        self.latent_kernels = ModuleList(latent_kernels)
        self.num_tasks = num_tasks
        self.num_latent = num_latent

    def forward(self, x, _, i=None, last_dim_is_batch=False, **params):
        cov = torch.zeros((len(x), len(x)))

        for d in range(self.num_tasks):
            for dprime in range(d, self.num_tasks):
                d_mask = i == d
                d_prime_mask = i == dprime

                x1 = x[d_mask]
                x2 = x[d_prime_mask]
                latent = [
                    b[d] * b[dprime] * k(x1=x1, x2=x2)
                    for b, k in zip(self.kernel_coefficients,
                                    self.latent_kernels)
                ]

                pair_cov = torch.sum(torch.stack(latent), dim=0)

                task_mask = d_mask @ d_prime_mask.t()
                cov[task_mask] = pair_cov.flatten()
                cov[task_mask.t()] = pair_cov.t().flatten()

        return cov + (torch.eye(len(x)) * 1e-4)
