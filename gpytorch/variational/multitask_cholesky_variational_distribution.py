#!/usr/bin/env python3

import torch
from ..lazy import CholLazyTensor
from ..distributions import MultivariateNormal
from .variational_distribution import VariationalDistribution


class MultitaskCholeskyVariationalDistribution(VariationalDistribution):
    def __init__(
        self, num_inducing_points, num_tasks, init_mean_value=3.,
    ):
        super().__init__(num_inducing_points)
        self.num_inducing_points = num_inducing_points
        self.num_tasks = num_tasks

        # Variational mean
        self.register_parameter("variational_mean", torch.nn.Parameter(torch.full(
            (self.num_tasks, self.num_inducing_points), fill_value=-(init_mean_value / (self.num_tasks - 1))
        )))
        num_inducing_per_task = self.num_inducing_points // self.num_tasks
        for i in range(self.num_tasks):
            self.variational_mean.data[i, i * num_inducing_per_task : (i + 1) * num_inducing_per_task] = init_mean_value

        # Variational covariance
        self.register_parameter("chol_variational_covar", torch.nn.Parameter(torch.eye(self.num_inducing_points)))

    def initialize_variational_distribution(self, prior_dist):
        prior_covar = prior_dist.covariance_matrix.detach()
        while prior_covar.dim() > 2:
            prior_covar = prior_covar[0]
        self.chol_variational_covar.data.copy_(torch.matmul(
            prior_covar.double().inverse().cholesky().type_as(prior_dist.mean),
            self.chol_variational_covar.data
        ))
        pass

    @property
    def variational_distribution(self):
        """
        Return the variational distribution q(u) that this module represents.
        In this simplest case, this involves directly returning the variational mean. For the variational covariance
        matrix, we consider the lower triangle of the registered variational covariance parameter, while also ensuring
        that the diagonal remains positive.
        """
        variational_mean = self.variational_mean
        chol_variational_covar = self.chol_variational_covar
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        # First make the cholesky factor is upper triangular
        # And has a positive diagonal
        strictly_lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar = chol_variational_covar.mul(strictly_lower_mask)
        variational_covar = CholLazyTensor(chol_variational_covar).repeat(self.num_tasks, 1, 1)
        return MultivariateNormal(variational_mean, variational_covar)
