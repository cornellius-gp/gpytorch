from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from .. import settings
from ..functions import add_diag
from ..lazy import (
    BlockDiagLazyTensor,
    DiagLazyTensor,
    KroneckerProductLazyTensor,
    MatmulLazyTensor,
    NonLazyTensor,
    RootLazyTensor,
)
from ..likelihoods import _GaussianLikelihoodBase, Likelihood
from .noise_models import MultitaskHomoskedasticNoise


def _eval_covar_matrix(task_noise_covar_factor, log_noise):
    num_tasks = task_noise_covar_factor.size(0)
    return task_noise_covar_factor.matmul(task_noise_covar_factor.transpose(-1, -2)) + log_noise.exp() * torch.eye(
        num_tasks
    )


def _eval_corr_matrix(task_noise_corr_factor):
    M = task_noise_corr_factor.matmul(task_noise_corr_factor.transpose(-1, -2))
    dsqrtinv = 1 / M.diag().sqrt()
    return M * dsqrtinv.unsqueeze(-1).matmul(dsqrtinv.unsqueeze(0))


class _MultitaskGaussianLikelihoodBase(_GaussianLikelihoodBase):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `log_noise` parameter for each task.)

    Like the Gaussian likelihood, this object can be used with exact inference.
    """

    def __init__(self, num_tasks, log_noise_covar, rank=0, task_correlation_prior=None, batch_size=1):
        """
        Args:
            num_tasks (int): Number of tasks.

            log_noise_covar (:obj:`gpytorch.module.Module`): A model for the log-noise covariance. This can be a
            simple homoskedastic model, or a GP that is to be fitted on the observed measurement errors.

            rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
            then a diagonal covariance matrix is fit.

            task_correlation_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise correlation matrix.
            Only used when `rank` > 0.

            batch_size (int): Number of batches.

        """
        super(_MultitaskGaussianLikelihoodBase, self).__init__(log_noise_covar=log_noise_covar)
        if rank != 0:
            self.register_parameter(
                name="task_noise_corr_factor", parameter=torch.nn.Parameter(torch.randn(batch_size, num_tasks, rank))
            )
            if task_correlation_prior is not None:
                self.register_derived_prior(
                    name="MultitaskErrorCorrelationPrior",
                    prior=task_correlation_prior,
                    parameter_names=("task_noise_corr_factor",),
                    transform=_eval_corr_matrix,
                )
        elif task_correlation_prior is not None:
            raise ValueError("Can only specify task_correlation_prior if rank>0")
        self.num_tasks = num_tasks

    def forward(self, input, *params):
        """
        Adds the log task noises to the diagonal of the covariance matrix of the supplied
        :obj:`gpytorch.distributions.MultivariateNormal` or
        :obj:`gpytorch.distributions.MultitaskMultivariateNormal`, in case of
        `rank` == 0. Otherwise, adds a rank `rank` covariance matrix to it.

        To accomplish this, we form a new :obj:`gpytorch.lazy.KroneckerProductLazyTensor` between :math:`I_{n}`,
        an identity matrix with size equal to the data and a (not necessarily diagonal) matrix containing the task
        noises :math:`D_{t}`.

        We also incorporate a shared `log_noise` parameter from the base
        :class:`gpytorch.likelihoods.GaussianLikelihood` that we extend.

        The final covariance matrix after this method is then :math:`K + D_{t} \otimes I_{n} + \sigma^{2}I_{nt}`.

        Args:
            input (:obj:`gpytorch.distributions.MultitaskMultivariateNormal`): Random variable whose covariance
                matrix is a :obj:`gpytorch.lazy.LazyTensor` we intend to augment.
        Returns:
            :obj:`gpytorch.distributions.MultitaskMultivariateNormal`: A new random variable whose covariance
            matrix is a :obj:`gpytorch.lazy.LazyTensor` with :math:`D_{t} \otimes I_{n}` and :math:`\sigma^{2}I_{nt}`
            added.
        """
        mean, covar = input.mean, input.lazy_covariance_matrix
        batch_shape, n = covar.shape[:-2], covar.shape[-1] // self.num_tasks

        if hasattr(self, "task_noise_corr_factor"):
            task_noise_corr_factor = self.task_noise_corr_factor
            if len(batch_shape) > 0:
                if settings.debug.on() and task_noise_corr_factor.size(0) > 1:
                    raise RuntimeError(
                        "With batch_size > 1, expected a batched MultitaskMultivariateNormal distribution."
                    )
                task_noise_corr_factor = task_noise_corr_factor.squeeze(0)
            # TODO: This is inefficient, find a better way to do this
            task_corr = NonLazyTensor(_eval_corr_matrix(task_noise_corr_factor))
        else:
            task_corr = DiagLazyTensor(
                torch.ones(batch_shape + torch.Size([self.num_tasks]), dtype=covar.dtype, device=covar.device)
            )

        log_noise_covar = self.log_noise_covar(*params)
        D_sem = log_noise_covar.exp().sqrt()
        task_covar_blocks = MatmulLazyTensor(MatmulLazyTensor(D_sem, task_corr.repeat(n, 1, 1)), D_sem)
        task_covar = BlockDiagLazyTensor(task_covar_blocks)
        return input.__class__(mean, covar + task_covar)

    def variational_log_probability(self, input, target):
        raise NotImplementedError("Variational inference with Multitask Gaussian likelihood is not yet supported")


class MultitaskGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `log_noise` parameter for each task.)

    Like the Gaussian likelihood, this object can be used with exact inference.
    """

    def __init__(self, num_tasks, rank=0, task_correlation_prior=None, batch_size=1, log_noise_prior=None):
        """
        Args:
            num_tasks (int): Number of tasks.

            rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
            then a diagonal covariance matrix is fit.

            task_correlation_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise correlaton matrix.
            Only used when `rank` > 0.

        """
        log_noise_covar = MultitaskHomoskedasticNoise(
            num_tasks=num_tasks, log_noise_prior=log_noise_prior, batch_size=1
        )
        super(MultitaskGaussianLikelihood, self).__init__(
            num_tasks=num_tasks,
            log_noise_covar=log_noise_covar,
            rank=rank,
            task_correlation_prior=task_correlation_prior,
            batch_size=batch_size,
        )


class MultitaskGaussianLikelihood_Kronecker(_MultitaskGaussianLikelihoodBase):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `log_noise` parameter for each task.)

    Like the Gaussian likelihood, this object can be used with exact inference.
    """

    def __init__(self, num_tasks, rank=0, task_prior=None, batch_size=1, log_noise_prior=None):
        """
        Args:
            num_tasks (int): Number of tasks.

            rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
            then a diagonal covariance matrix is fit.

            task_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise covariance matrix if
            `rank` > 0, or a prior over the log of just the diagonal elements, if `rank` == 0.

        """
        Likelihood.__init__(self)
        self.register_parameter(
            name="log_noise", parameter=torch.nn.Parameter(torch.zeros(batch_size, 1)), prior=log_noise_prior
        )
        if rank == 0:
            self.register_parameter(
                name="log_task_noises",
                parameter=torch.nn.Parameter(torch.zeros(batch_size, num_tasks)),
                prior=task_prior,
            )
        else:
            self.register_parameter(
                name="task_noise_covar_factor", parameter=torch.nn.Parameter(torch.randn(batch_size, num_tasks, rank))
            )
            if task_prior is not None:
                self.register_derived_prior(
                    name="MultitaskErrorCovariancePrior",
                    prior=task_prior,
                    parameter_names=("task_noise_covar_factor", "log_noise"),
                    transform=_eval_covar_matrix,
                )
        self.num_tasks = num_tasks

    def forward(self, input, *params):
        """
        Adds the log task noises to the diagonal of the covariance matrix of the supplied
        :obj:`gpytorch.distributions.MultivariateNormal` or
        :obj:`gpytorch.distributions.MultitaskMultivariateNormal`, in case of
        `rank` == 0. Otherwise, adds a rank `rank` covariance matrix to it.

        To accomplish this, we form a new :obj:`gpytorch.lazy.KroneckerProductLazyTensor` between :math:`I_{n}`,
        an identity matrix with size equal to the data and a (not necessarily diagonal) matrix containing the task
        noises :math:`D_{t}`.

        We also incorporate a shared `log_noise` parameter from the base
        :class:`gpytorch.likelihoods.GaussianLikelihood` that we extend.

        The final covariance matrix after this method is then :math:`K + D_{t} \otimes I_{n} + \sigma^{2}I_{nt}`.

        Args:
            input (:obj:`gpytorch.distributions.MultitaskMultivariateNormal`): Random variable whose covariance
                matrix is a :obj:`gpytorch.lazy.LazyTensor` we intend to augment.
        Returns:
            :obj:`gpytorch.distributions.MultitaskMultivariateNormal`: A new random variable whose covariance
            matrix is a :obj:`gpytorch.lazy.LazyTensor` with :math:`D_{t} \otimes I_{n}` and :math:`\sigma^{2}I_{nt}`
            added.
        """
        mean, covar = input.mean, input.lazy_covariance_matrix

        if hasattr(self, "log_task_noises"):
            noises = self.log_task_noises.exp()
            if covar.ndimension() == 2:
                if settings.debug.on() and noises.size(0) > 1:
                    raise RuntimeError(
                        "With batch_size > 1, expected a batched MultitaskMultivariateNormal distribution."
                    )
                noises = noises.squeeze(0)
            task_var_lt = DiagLazyTensor(noises)
        else:
            task_noise_covar_factor = self.task_noise_covar_factor
            if covar.ndimension() == 2:
                if settings.debug.on() and task_noise_covar_factor.size(0) > 1:
                    raise RuntimeError(
                        "With batch_size > 1, expected a batched MultitaskMultivariateNormal distribution."
                    )
                task_noise_covar_factor = task_noise_covar_factor.squeeze(0)
            task_var_lt = RootLazyTensor(task_noise_covar_factor)

        device = (
            self.log_task_noises.device if hasattr(self, "log_task_noises") else self.task_noise_covar_factor.device
        )

        if covar.ndimension() == 2:
            eye_lt = DiagLazyTensor(torch.ones(covar.size(-1) // self.num_tasks, device=device))
        else:
            eye_lt = DiagLazyTensor(torch.ones(covar.size(0), covar.size(-1) // self.num_tasks, device=device))
            # Make sure the batch sizes are going to match
            if task_var_lt.size(0) == 1:
                task_var_lt = task_var_lt.repeat(eye_lt.size(0), 1, 1)

        covar_kron_lt = KroneckerProductLazyTensor(task_var_lt, eye_lt)
        covar = covar + covar_kron_lt

        noise = self.log_noise.exp()
        if covar.ndimension() == 2:
            if settings.debug.on() and noise.size(0) > 1:
                raise RuntimeError("With batch_size > 1, expected a batched MultitaskMultivariateNormal distribution.")
            noise = noise.squeeze(0)

        covar = add_diag(covar, noise)
        return input.__class__(mean, covar)
