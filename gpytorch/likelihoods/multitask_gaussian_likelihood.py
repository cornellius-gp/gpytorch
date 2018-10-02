from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from ..functions import add_diag
from ..lazy import DiagLazyTensor, KroneckerProductLazyTensor, RootLazyTensor
from ..likelihoods import GaussianLikelihood
from .. import settings


def _eval_covar_matrix(task_noise_covar_factor, log_noise):
    num_tasks = task_noise_covar_factor.size(0)
    return task_noise_covar_factor.matmul(task_noise_covar_factor.transpose(-1, -2)) + log_noise.exp() * torch.eye(
        num_tasks
    )


class MultitaskGaussianLikelihood(GaussianLikelihood):
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
        super(MultitaskGaussianLikelihood, self).__init__(batch_size=batch_size, log_noise_prior=log_noise_prior)

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

    def forward(self, input):
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

        if covar.ndimension() == 2:
            eye_lt = DiagLazyTensor(torch.ones(covar.size(-1) // self.num_tasks, device=self.log_noise.device))
        else:
            eye_lt = DiagLazyTensor(
                torch.ones(covar.size(0), covar.size(-1) // self.num_tasks, device=self.log_noise.device)
            )
            # Make sure the batch sizes are going to match
            if task_var_lt.size(0) == 1:
                task_var_lt = task_var_lt.repeat(eye_lt.size(0), 1, 1)

        covar_kron_lt = KroneckerProductLazyTensor(task_var_lt, eye_lt)
        covar = covar + covar_kron_lt

        noise = self.noise
        if covar.ndimension() == 2:
            if settings.debug.on() and noise.size(0) > 1:
                raise RuntimeError("With batch_size > 1, expected a batched MultitaskMultivariateNormal distribution.")
            noise = noise.squeeze(0)

        covar = add_diag(covar, noise)
        return input.__class__(mean, covar)

    def variational_log_probability(self, input, target):
        raise NotImplementedError("Variational inference with Multitask Gaussian likelihood is not yet supported")
