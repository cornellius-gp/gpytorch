from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from gpytorch.functions import add_diag
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.lazy import DiagLazyVariable, KroneckerProductLazyVariable


class MultitaskGaussianLikelihood(GaussianLikelihood):
    def __init__(self, n_tasks, log_shared_noise_prior=None, log_task_noises_prior=None):
        # TODO: Remove deprecated log_noise_bounds kwarg
        super(MultitaskGaussianLikelihood, self).__init__()
        self.register_parameter(
            name="log_task_noises", parameter=torch.nn.Parameter(torch.zeros(n_tasks)), prior=log_task_noises_prior
        )
        self.n_tasks = n_tasks

    def forward(self, input):
        mean, covar = input.representation()
        eye_lv = DiagLazyVariable(torch.ones(int(covar.size(-1) / self.n_tasks), device=self.log_noise.device))
        task_var_lv = DiagLazyVariable(self.log_task_noises.exp())
        diag_kron_lv = KroneckerProductLazyVariable(task_var_lv, eye_lv)
        noise = covar + diag_kron_lv
        noise = add_diag(noise, self.log_noise.exp())
        return input.__class__(mean, noise)

    def log_probability(self, input, target):
        raise NotImplementedError("Variational inference with Multitask Gaussian likelihood is not yet supported")
