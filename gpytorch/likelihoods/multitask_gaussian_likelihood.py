from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from gpytorch.functions import add_diag
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.lazy import DiagLazyVariable, KroneckerProductLazyVariable


class MultitaskGaussianLikelihood(GaussianLikelihood):
    """
    A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a different `log_noise` parameter for each task. Like the Gaussian likelihood, this object can be used with
    exact inference.
    """
    def __init__(self, n_tasks, log_task_noises_prior=None):
        """
        Args:
            n_tasks (int): Number of tasks. This likelihood will create a log noise parameter for each task.
            log_task_noises_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the log task noises.
        """
        super(MultitaskGaussianLikelihood, self).__init__()
        self.register_parameter(
            name="log_task_noises",
            parameter=torch.nn.Parameter(torch.zeros(n_tasks)),
            prior=log_task_noises_prior,
        )
        self.n_tasks = n_tasks

    def forward(self, input):
        """
        Adds the log task noises to the diagonal of the covariance matrix of the supplied
        :obj:`gpytorch.random_variables.GaussianRandomVariable` or
        :obj:`gpytorch.random_variables.MultitaskGaussianRandomVariable`.

        To accomplish this, we form a new :obj:`gpytorch.lazy.KroneckerProductLazyVariable` between :math:`I_{n}`,
        an identity matrix with size equal to the data and a diagonal matrix containing the task noises :math:`D_{t}`.

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
        task_var_lv = DiagLazyVariable(self.log_task_noises.exp())
        diag_kron_lv = KroneckerProductLazyVariable(task_var_lv, eye_lv)
        noise = covar + diag_kron_lv
        noise = add_diag(noise, self.log_noise.exp())
        return input.__class__(mean, noise)

    def log_probability(self, input, target):
        raise NotImplementedError("Variational inference with Multitask Gaussian likelihood is not yet supported")
