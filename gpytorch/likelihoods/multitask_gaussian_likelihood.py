#!/usr/bin/env python3

import warnings
from typing import Any

import torch
from torch import Tensor

from ..constraints import GreaterThan
from ..distributions import base_distributions
from ..functions import add_diag
from ..lazy import (
    BlockDiagLazyTensor,
    DiagLazyTensor,
    KroneckerProductLazyTensor,
    MatmulLazyTensor,
    RootLazyTensor,
    lazify,
)
from ..likelihoods import Likelihood, _GaussianLikelihoodBase
from ..utils.warnings import OldVersionWarning
from .noise_models import MultitaskHomoskedasticNoise


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
                    "MultitaskErrorCorrelationPrior", task_correlation_prior, lambda: self._eval_corr_matrix
                )
        elif task_correlation_prior is not None:
            raise ValueError("Can only specify task_correlation_prior if rank>0")
        self.num_tasks = num_tasks
        self.rank = rank
        # Handle deprecation of parameterization - TODO: Remove in future release
        self._register_load_state_dict_pre_hook(deprecate_task_noise_corr)

    def _eval_corr_matrix(self):
        tnc = self.task_noise_corr
        fac_diag = torch.ones(*tnc.shape[:-1], self.num_tasks, device=tnc.device, dtype=tnc.dtype)
        Cfac = torch.diag_embed(fac_diag)
        Cfac[..., self.tidcs[0], self.tidcs[1]] = self.task_noise_corr
        # squared rows must sum to one for this to be a correlation matrix
        C = Cfac / Cfac.pow(2).sum(dim=-1, keepdim=True).sqrt()
        return C @ C.transpose(-1, -2)

    def _shaped_noise_covar(self, base_shape, *params):
        if len(base_shape) >= 2:
            *batch_shape, n, _ = base_shape
        else:
            *batch_shape, n = base_shape

        # compute the noise covariance
        if len(params) > 0:
            shape = None
        else:
            shape = base_shape if len(base_shape) == 1 else base_shape[:-1]
        noise_covar = self.noise_covar(*params, shape=shape)

        if self.rank > 0:
            # if rank > 0, compute the task correlation matrix
            # TODO: This is inefficient, change repeat so it can repeat LazyTensors w/ multiple batch dimensions
            task_corr = self._eval_corr_matrix()
            exp_shape = torch.Size([*batch_shape, n]) + task_corr.shape[-2:]
            task_corr_exp = lazify(task_corr.unsqueeze(-3).expand(exp_shape))
            noise_sem = noise_covar.sqrt()
            task_covar_blocks = MatmulLazyTensor(MatmulLazyTensor(noise_sem, task_corr_exp), noise_sem)
        else:
            # otherwise tasks are uncorrelated
            if isinstance(noise_covar, DiagLazyTensor):
                flattened_diag = noise_covar._diag.view(*noise_covar._diag.shape[:-2], -1)
                return DiagLazyTensor(flattened_diag)
            task_covar_blocks = noise_covar
        if len(batch_shape) == 1:
            # TODO: Properly support general batch shapes in BlockDiagLazyTensor (no shape arithmetic)
            tcb_eval = task_covar_blocks.evaluate()
            task_covar = BlockDiagLazyTensor(lazify(tcb_eval), block_dim=-3)
        else:
            task_covar = BlockDiagLazyTensor(task_covar_blocks)

        return task_covar

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> base_distributions.Normal:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
        noise = noise.view(*noise.shape[:-1], *function_samples.shape[-2:])
        return base_distributions.Independent(base_distributions.Normal(function_samples, noise.sqrt()), 1)


class MultitaskGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `log_noise` parameter for each task.). This likelihood assumes homoskedastic noise.

    Like the Gaussian likelihood, this object can be used with exact inference.
    """

    def __init__(
        self,
        num_tasks,
        rank=0,
        task_correlation_prior=None,
        batch_shape=torch.Size(),
        noise_prior=None,
        noise_constraint=None,
    ):
        """
        Args:
            num_tasks (int): Number of tasks.

            rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
            then a diagonal covariance matrix is fit.

            task_correlation_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise correlaton matrix.
            Only used when `rank` > 0.

        """
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        noise_covar = MultitaskHomoskedasticNoise(
            num_tasks=num_tasks, noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        super().__init__(
            num_tasks=num_tasks,
            noise_covar=noise_covar,
            rank=rank,
            task_correlation_prior=task_correlation_prior,
            batch_shape=batch_shape,
        )

        self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        self.register_constraint("raw_noise", noise_constraint)

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _shaped_noise_covar(self, base_shape, *params):
        noise_covar = super()._shaped_noise_covar(base_shape, *params)
        noise = self.noise
        return noise_covar.add_diag(noise)


class MultitaskGaussianLikelihoodKronecker(_MultitaskGaussianLikelihoodBase):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `noise` parameter for each task.)

    Like the Gaussian likelihood, this object can be used with exact inference.

    Note: This Likelihood is scheduled to be deprecated and replaced by an improved version of
    `MultitaskGaussianLikelihood`. Use this only for compatibility with batched Multitask models.
    """

    def __init__(
        self, num_tasks, rank=0, task_prior=None, batch_shape=torch.Size(), noise_prior=None, noise_constraint=None
    ):
        """
        Args:
            num_tasks (int): Number of tasks.

            rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
            then a diagonal covariance matrix is fit.

            task_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise covariance matrix if
            `rank` > 0, or a prior over the log of just the diagonal elements, if `rank` == 0.

        """
        super(Likelihood, self).__init__()
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)
        self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if rank == 0:
            self.register_parameter(
                name="raw_task_noises", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks))
            )
            if task_prior is not None:
                raise RuntimeError("Cannot set a `task_prior` if rank=0")
        else:
            self.register_parameter(
                name="task_noise_covar_factor", parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank))
            )
            if task_prior is not None:
                self.register_prior("MultitaskErrorCovariancePrior", task_prior, self._eval_covar_matrix)
        self.num_tasks = num_tasks
        self.rank = rank

        self.register_constraint("raw_noise", noise_constraint)

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        covar_factor = self.task_noise_covar_factor
        noise = self.noise
        D = noise * torch.eye(self.num_tasks, dtype=noise.dtype, device=noise.device)
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D

    def marginal(self, function_dist, *params, **kwargs):
        r"""
        Adds the task noises to the diagonal of the covariance matrix of the supplied
        :obj:`gpytorch.distributions.MultivariateNormal` or :obj:`gpytorch.distributions.MultitaskMultivariateNormal`,
        in case of `rank` == 0. Otherwise, adds a rank `rank` covariance matrix to it.

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

        if self.rank == 0:
            task_noises = self.raw_noise_constraint.transform(self.raw_task_noises)
            task_var_lt = DiagLazyTensor(task_noises)
            dtype, device = task_noises.dtype, task_noises.device
        else:
            task_noise_covar_factor = self.task_noise_covar_factor
            task_var_lt = RootLazyTensor(task_noise_covar_factor)
            dtype, device = task_noise_covar_factor.dtype, task_noise_covar_factor.device

        eye_lt = DiagLazyTensor(
            torch.ones(*covar.batch_shape, covar.size(-1) // self.num_tasks, dtype=dtype, device=device)
        )
        task_var_lt = task_var_lt.expand(*covar.batch_shape, *task_var_lt.matrix_shape)

        covar_kron_lt = KroneckerProductLazyTensor(eye_lt, task_var_lt)
        covar = covar + covar_kron_lt

        noise = self.noise
        covar = add_diag(covar, noise)
        return function_dist.__class__(mean, covar)


def deprecate_task_noise_corr(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + "task_noise_corr_factor" in state_dict:
        # Remove after 1.0
        warnings.warn(
            "Loading a deprecated parameterization of _MultitaskGaussianLikelihoodBase. Consider re-saving your model.",
            OldVersionWarning,
        )
        # construct the task correlation matrix from the factors using the old parameterization
        corr_factor = state_dict.pop(prefix + "task_noise_corr_factor").squeeze(0)
        corr_diag = state_dict.pop(prefix + "task_noise_corr_diag").squeeze(0)
        num_tasks, rank = corr_factor.shape[-2:]
        M = corr_factor.matmul(corr_factor.transpose(-1, -2))
        idx = torch.arange(M.shape[-1], dtype=torch.long, device=M.device)
        M[..., idx, idx] += corr_diag
        sem_inv = 1 / torch.diagonal(M, dim1=-2, dim2=-1).sqrt().unsqueeze(-1)
        C = M * sem_inv.matmul(sem_inv.transpose(-1, -2))
        # perform a Cholesky decomposition and extract the required entries
        L = torch.cholesky(C)
        tidcs = torch.tril_indices(num_tasks, rank)[:, 1:]
        task_noise_corr = L[..., tidcs[0], tidcs[1]]
        state_dict[prefix + "task_noise_corr"] = task_noise_corr
