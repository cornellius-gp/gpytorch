#!/usr/bin/env python3

from typing import Any

import torch
from torch import Tensor

from ..constraints import GreaterThan
from ..distributions import base_distributions
from ..lazy import (
    ConstantDiagLazyTensor,
    DiagLazyTensor,
    KroneckerProductDiagLazyTensor,
    KroneckerProductLazyTensor,
    RootLazyTensor,
)
from ..likelihoods import Likelihood, _GaussianLikelihoodBase
from ..utils.pivoted_cholesky import pivoted_cholesky


class _MultitaskGaussianLikelihoodBase(_GaussianLikelihoodBase):
    """Base class for multi-task Gaussian Likelihoods, supporting general heteroskedastic noise models. """

    def __init__(self, num_tasks, noise_covar, rank=0, task_correlation_prior=None, batch_shape=torch.Size()):
        """
        Args:
            num_tasks (int):
                Number of tasks.
            noise_covar (:obj:`gpytorch.module.Module`):
                A model for the noise covariance. This can be a simple homoskedastic noise model, or a GP
                that is to be fitted on the observed measurement errors.
            rank (int):
                The rank of the task noise covariance matrix to fit. If `rank` is set to 0, then a diagonal covariance
                matrix is fit.
            task_correlation_prior (:obj:`gpytorch.priors.Prior`):
                Prior to use over the task noise correlation matrix. Only used when `rank` > 0.
            batch_shape (torch.Size):
                Number of batches.
        """
        super().__init__(noise_covar=noise_covar)
        if rank != 0:
            if rank > num_tasks:
                raise ValueError(f"Cannot have rank ({rank}) greater than num_tasks ({num_tasks})")
            tidcs = torch.tril_indices(num_tasks, rank, dtype=torch.long)
            self.tidcs = tidcs[:, 1:]  # (1, 1) must be 1.0, no need to parameterize this
            task_noise_corr = torch.randn(*batch_shape, self.tidcs.size(-1))
            self.register_parameter("task_noise_corr", torch.nn.Parameter(task_noise_corr))
            if task_correlation_prior is not None:
                self.register_prior(
                    "MultitaskErrorCorrelationPrior", task_correlation_prior, lambda m: m._eval_corr_matrix
                )
        elif task_correlation_prior is not None:
            raise ValueError("Can only specify task_correlation_prior if rank>0")
        self.num_tasks = num_tasks
        self.rank = rank

    def _eval_corr_matrix(self):
        tnc = self.task_noise_corr
        fac_diag = torch.ones(*tnc.shape[:-1], self.num_tasks, device=tnc.device, dtype=tnc.dtype)
        Cfac = torch.diag_embed(fac_diag)
        Cfac[..., self.tidcs[0], self.tidcs[1]] = self.task_noise_corr
        # squared rows must sum to one for this to be a correlation matrix
        C = Cfac / Cfac.pow(2).sum(dim=-1, keepdim=True).sqrt()
        return C @ C.transpose(-1, -2)

    def marginal(self, function_dist, *params, **kwargs):
        r"""
        If `rank` == 0, adds the task noises to the diagonal of the covariance matrix of the supplied
        :obj:`gpytorch.distributions.MultivariateNormal` or :obj:`gpytorch.distributions.MultitaskMultivariateNormal`.
        Otherwise, adds a rank `rank` covariance matrix to it.

        To accomplish this, we form a new :obj:`gpytorch.lazy.KroneckerProductLazyTensor` between :math:`I_{n}`,
        an identity matrix with size equal to the data and a (not necessarily diagonal) matrix containing the task
        noises :math:`D_{t}`.

        We also incorporate a shared `noise` parameter from the base
        :class:`gpytorch.likelihoods.GaussianLikelihood` that we extend.

        The final covariance matrix after this method is then :math:`K + D_{t} \otimes I_{n} + \sigma^{2}I_{nt}`.

        Args:
            function_dist (:obj:`gpytorch.distributions.MultitaskMultivariateNormal`): Random variable whose covariance
                matrix is a :obj:`gpytorch.lazy.LazyTensor` we intend to augment.
        Returns:
            :obj:`gpytorch.distributions.MultitaskMultivariateNormal`: A new random variable whose covariance
            matrix is a :obj:`gpytorch.lazy.LazyTensor` with :math:`D_{t} \otimes I_{n}` and :math:`\sigma^{2}I_{nt}`
            added.
        """
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

        covar_kron_lt = self._shaped_noise_covar(mean.shape, add_noise=self.has_global_noise)
        covar = covar + covar_kron_lt

        return function_dist.__class__(mean, covar)

    def _shaped_noise_covar(self, shape, add_noise=True, *params, **kwargs):
        if not self.has_task_noise:
            noise = ConstantDiagLazyTensor(self.noise, diag_shape=shape[-2] * self.num_tasks)
            return noise

        if self.rank == 0:
            task_noises = self.raw_task_noises_constraint.transform(self.raw_task_noises)
            task_var_lt = DiagLazyTensor(task_noises)
            dtype, device = task_noises.dtype, task_noises.device
            ckl_init = KroneckerProductDiagLazyTensor
        else:
            task_noise_covar_factor = self.task_noise_covar_factor
            task_var_lt = RootLazyTensor(task_noise_covar_factor)
            dtype, device = task_noise_covar_factor.dtype, task_noise_covar_factor.device
            ckl_init = KroneckerProductLazyTensor

        eye_lt = ConstantDiagLazyTensor(torch.ones(*shape[:-2], 1, dtype=dtype, device=device), diag_shape=shape[-2])
        task_var_lt = task_var_lt.expand(*shape[:-2], *task_var_lt.matrix_shape)

        # to add the latent noise we exploit the fact that
        # I \kron D_T + \sigma^2 I_{NT} = I \kron (D_T + \sigma^2 I)
        # which allows us to move the latent noise inside the task dependent noise
        # thereby allowing exploitation of Kronecker structure in this likelihood.
        if add_noise and self.has_global_noise:
            noise = ConstantDiagLazyTensor(self.noise, diag_shape=task_var_lt.shape[-1])
            task_var_lt = task_var_lt + noise

        covar_kron_lt = ckl_init(eye_lt, task_var_lt)

        return covar_kron_lt

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> base_distributions.Normal:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
        noise = noise.view(*noise.shape[:-1], *function_samples.shape[-2:])
        return base_distributions.Independent(base_distributions.Normal(function_samples, noise.sqrt()), 1)


class MultitaskGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `noise` parameter for each task.)

    Like the Gaussian likelihood, this object can be used with exact inference.

    """

    def __init__(
        self,
        num_tasks,
        rank=0,
        task_prior=None,
        batch_shape=torch.Size(),
        noise_prior=None,
        noise_constraint=None,
        has_global_noise=True,
        has_task_noise=True,
    ):
        """
        Args:
            num_tasks (int): Number of tasks.

            rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
            then a diagonal covariance matrix is fit.

            task_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise covariance matrix if
            `rank` > 0, or a prior over the log of just the diagonal elements, if `rank` == 0.

            has_global_noise (bool): whether to include a \\sigma^2 I_{nt} term in the noise model.

            has_task_noise (bool): whether to include task-specific noise terms, which add I_n \kron D_T
            into the noise model.

            At least one of has_global_noise or has_task_noise should be specified.

        """
        super(Likelihood, self).__init__()
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        if not has_task_noise and not has_global_noise:
            raise ValueError(
                "At least one of has_task_noise or has_global_noise must be specified. "
                "Attempting to specify a likelihood that has no noise terms."
            )

        if has_task_noise:
            if rank == 0:
                self.register_parameter(
                    name="raw_task_noises", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks))
                )
                self.register_constraint("raw_task_noises", noise_constraint)
                if noise_prior is not None:
                    self.register_prior("raw_task_noises_prior", noise_prior, lambda m: m.task_noises)
                if task_prior is not None:
                    raise RuntimeError("Cannot set a `task_prior` if rank=0")
            else:
                self.register_parameter(
                    name="task_noise_covar_factor",
                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank)),
                )
                if task_prior is not None:
                    self.register_prior("MultitaskErrorCovariancePrior", task_prior, lambda m: m._eval_covar_matrix)
        self.num_tasks = num_tasks
        self.rank = rank

        if has_global_noise:
            self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
            self.register_constraint("raw_noise", noise_constraint)
            if noise_prior is not None:
                self.register_prior("raw_noise_prior", noise_prior, lambda m: m.noise)

        self.has_global_noise = has_global_noise
        self.has_task_noise = has_task_noise

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    @property
    def task_noises(self):
        if self.rank == 0:
            return self.raw_task_noises_constraint.transform(self.raw_task_noises)
        else:
            raise AttributeError("Cannot set diagonal task noises when covariance has ", self.rank, ">0")

    @task_noises.setter
    def task_noises(self, value):
        if self.rank == 0:
            self._set_task_noises(value)
        else:
            raise AttributeError("Cannot set diagonal task noises when covariance has ", self.rank, ">0")

    def _set_noise(self, value):
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _set_task_noises(self, value):
        self.initialize(raw_task_noises=self.raw_task_noises_constraint.inverse_transform(value))

    @property
    def task_noise_covar(self):
        if self.rank > 0:
            return self.task_noise_covar_factor.matmul(self.task_noise_covar_factor.transpose(-1, -2))
        else:
            raise AttributeError("Cannot retrieve task noises when covariance is diagonal.")

    @task_noise_covar.setter
    def task_noise_covar(self, value):
        # internally uses a pivoted cholesky decomposition to construct a low rank
        # approximation of the covariance
        if self.rank > 0:
            self.task_noise_covar_factor.data = pivoted_cholesky(value, max_iter=self.rank)
        else:
            raise AttributeError("Cannot set non-diagonal task noises when covariance is diagonal.")

    def _eval_covar_matrix(self):
        covar_factor = self.task_noise_covar_factor
        noise = self.noise
        D = noise * torch.eye(self.num_tasks, dtype=noise.dtype, device=noise.device)
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D
