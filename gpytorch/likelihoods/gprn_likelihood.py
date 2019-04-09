import torch
import math
from torch.nn import Parameter
from .likelihood import Likelihood
from ..constraints import Positive
from ..lazy import DiagLazyTensor
from ..distributions import MultivariateNormal


class GPRNLikelihood(Likelihood):
    def __init__(
        self,
        num_gps,
        num_outputs,
        f_noise_constraint=None,
        y_noise_constraint=None
    ):
        super(GPRNLikelihood, self).__init__()

        if f_noise_constraint is None:
            f_noise_constraint = Positive()

        if y_noise_constraint is None:
            y_noise_constraint = Positive()

        self.num_outputs = num_outputs
        self.num_gps = num_gps

        self.log_beta_Y = Parameter(torch.zeros(num_outputs))
        self.log_beta_GP = Parameter(torch.zeros(num_gps))

        # self.register_parameter(name="raw_f_noise", parameter=Parameter(torch.zeros(num_gps)))
        # self.register_parameter(name="raw_y_noise", parameter=Parameter(torch.zeros(num_outputs)))
        # self.register_constraint("raw_f_noise", f_noise_constraint)
        # self.register_constraint("raw_y_noise", y_noise_constraint)

    @property
    def f_noise(self):
        return self.raw_f_noise_constraint.transform(self.raw_f_noise)

    @property
    def y_noise(self):
        return self.raw_y_noise_constraint.transform(self.raw_y_noise)

    @f_noise.setter
    def f_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)

        self.initialize(raw_f_noise=self.raw_f_noise_constraint.inverse_transform(value))

    @y_noise.setter
    def y_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)

        self.initialize(raw_y_noise=self.raw_y_noise_constraint.inverse_transform(value))

    def forward(self, function_samples, *params, **kwargs):
        # Assume function_samples is the output of calling rsample() to get n_samples samples.
        f = function_samples.transpose(-1, -2)

        f_weights, f_gp = f[..., :-self.num_gps], f[..., -self.num_gps:]
        f_weights = f_weights.contiguous().view(f.size(0), f.size(1), self.num_outputs, self.num_gps)
        f_gp = f_gp + (-0.5 * self.log_beta_GP).exp() * torch.randn_like(f_gp)
        mean_y = torch.matmul(f_weights, f_gp.unsqueeze(-1)).squeeze(-1)
        y_noise = (-0.5 * self.log_beta_Y).exp().expand_as(mean_y)
        return torch.distributions.Normal(mean_y.transpose(-2, -1), y_noise.transpose(-2, -1))

    def expected_log_prob(self, target, latent_func):
        mean_f, variance_f = latent_func.mean.squeeze(-1).t(), latent_func.variance.squeeze(-1).t()
        mean_weights, mean_gp = mean_f[:, :-self.num_gps], mean_f[:, -self.num_gps:]
        var_weights, var_gp = variance_f[:, :-self.num_gps], variance_f[:, -self.num_gps:]
        full_var_gp = var_gp + (-self.log_beta_GP).exp()

        mean_weights = mean_weights.view(-1, self.num_outputs, self.num_gps)  # B O GP
        var_weights = var_weights.view(-1, self.num_outputs, self.num_gps)  # B O GP
        #                      B O GP        B GP 1
        mean_Y = torch.matmul(mean_weights, mean_gp.unsqueeze(-1)).squeeze(-1)  # B O
        var_Y1 = torch.matmul(mean_weights.pow(2.0), full_var_gp.unsqueeze(-1)).squeeze(-1)  # B O
        var_Y2 = torch.matmul(var_weights, mean_gp.pow(2.0).unsqueeze(-1)).squeeze(-1)  # B O
        var_Y3 = torch.matmul(var_weights, full_var_gp.unsqueeze(-1)).squeeze(-1)  # B O
        var_Y = var_Y1 + var_Y2 + var_Y3

        beta_Y = self.log_beta_Y.exp()

        ell1 = -0.5 * (beta_Y * (target - mean_Y).pow(2.0)).sum()
        ell2 = -0.5 * (beta_Y * var_Y).sum()
        ell3 = 0.5 * target.size(0) * (self.log_beta_Y - math.log(2.0 * math.pi)).sum()
        ell = ell1 + ell2 + ell3

        return ell
