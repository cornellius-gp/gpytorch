from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from gpytorch.functions import add_diag
from gpytorch.lazy import DiagLazyVariable, KroneckerProductLazyVariable, RootLazyVariable
from gpytorch.likelihoods import GaussianLikelihood


def _eval_covar_matrix(task_noise_covar_factor, log_noise):
    n_tasks = task_noise_covar_factor.size(0)
    return task_noise_covar_factor.matmul(task_noise_covar_factor.transpose(-1, -2)) + log_noise.exp() * torch.eye(
        n_tasks
    )


class MultitaskGaussianLikelihood(GaussianLikelihood):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `log_noise` parameter for each task.)

    Like the Gaussian likelihood, this object can be used with exact inference.
    """

    def __init__(self, n_tasks, rank=0, task_prior=None):
        """
        Args:
            n_tasks (int): Number of tasks.

            rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
            then a diagonal covariance matrix is fit.

            task_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise covariance matrix if
            `rank` > 0, or a prior over the log of just the diagonal elements, if `rank` == 0.

        """
        super(MultitaskGaussianLikelihood, self).__init__()

        if rank == 0:
            self.register_parameter(
                name="log_task_noises", parameter=torch.nn.Parameter(torch.zeros(n_tasks)), prior=task_prior
            )
        else:
            self.register_parameter(
                name="task_noise_covar_factor", parameter=torch.nn.Parameter(torch.randn([n_tasks, rank]))
            )
            if task_prior is not None:
                self.register_derived_prior(
                    name="MultitaskErrorCovariancePrior",
                    prior=task_prior,
                    parameter_names=("task_noise_covar_factor", "log_noise"),
                    transform=_eval_covar_matrix,
                )
        self.n_tasks = n_tasks

    def forward(self, input):
        """
        Adds the log task noises to the diagonal of the covariance matrix of the supplied
        :obj:`gpytorch.random_variables.GaussianRandomVariable` or
        :obj:`gpytorch.random_variables.MultitaskGaussianRandomVariable`, in case of
        `rank` == 0. Otherwise, adds a rank `rank` covariance matrix to it.

        To accomplish this, we form a new :obj:`gpytorch.lazy.KroneckerProductLazyVariable` between :math:`I_{n}`,
        an identity matrix with size equal to the data and a (not necessarily diagonal) matrix containing the task
        noises :math:`D_{t}`.

        We also incorporate a shared `log_noise` parameter from the base
        :class:`gpytorch.likelihoods.GaussianLikelihood` that we extend.

        The final covariance matrix after this method is then :math:`K + D_{t} \otimes I_{n} + \sigma^{2}I_{nt}`.

        Args:
            input (:obj:`gpytorch.random_variables.MultitaskGaussianRandomVariable`): Random variable whose covariance
                matrix is a :obj:`gpytorch.lazy.LazyVariable` we intend to augment.
        Returns:
            :obj:`gpytorch.random_variables.MultitaskGaussianRandomVariable`: A new random variable whose covariance
            matrix is a :obj:`gpytorch.lazy.LazyVariable` with :math:`D_{t} \otimes I_{n}` and :math:`\sigma^{2}I_{nt}`
            added.
        """
        mean, covar = input.representation()
        eye_lv = DiagLazyVariable(torch.ones(covar.size(-1) // self.n_tasks, device=self.log_noise.device))
        if hasattr(self, "log_task_noises"):
            task_var_lv = DiagLazyVariable(self.log_task_noises.exp())
        else:
            task_var_lv = RootLazyVariable(self.task_noise_covar_factor)
        covar_kron_lv = KroneckerProductLazyVariable(task_var_lv, eye_lv)
        noise = covar + covar_kron_lv
        noise = add_diag(noise, self.log_noise.exp())
        return input.__class__(mean, noise)

    def log_probability(self, input, target):
        raise NotImplementedError("Variational inference with Multitask Gaussian likelihood is not yet supported")
