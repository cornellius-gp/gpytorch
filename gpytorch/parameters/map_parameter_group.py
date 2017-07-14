import math
import torch
from .parameter_group import ParameterGroup
from .parameter_with_prior import ParameterWithPrior
from torch.autograd import Variable
from ..utils import pd_catcher, LBFGS


class MAPParameterGroup(ParameterGroup):
    def __init__(self, **kwargs):
        super(MAPParameterGroup, self).__init__()
        self._priors = {}

        for name, param_with_prior in kwargs.items():
            if not isinstance(param_with_prior, ParameterWithPrior):
                raise RuntimeError('All parameters in an MAPParameterGroup must be of type ParameterWithPrior')

            setattr(self, name, param_with_prior.param)
            self._priors[name] = param_with_prior.prior

        self._options = {
            'grad_tolerance': 1e-5,
            'relative_loss_tolerance': 1e-3,
            'optim_options': {
                'max_iter': 50
            },
        }

    def update(self, log_likelihood_closure):
        named_parameters = list(self.named_parameters())
        names = [kv[0] for kv in named_parameters]
        parameters = [kv[1] for kv in named_parameters]
        optim_options = self._options['optim_options']
        optimizer = LBFGS(parameters, line_search_fn='backtracking', **optim_options)
        optimizer.n_iter = 0

        @pd_catcher(catch_function=lambda: Variable(torch.Tensor([10000])))
        def step_closure():
            optimizer.zero_grad()
            optimizer.n_iter += 1
            loss = -log_likelihood_closure()
            for name, parameter in zip(names, parameters):
                loss -= self._priors[name].log_probability(parameter)
            loss.backward()
            return loss

        loss = optimizer.step(step_closure)
        self.previous_loss = loss.data.squeeze()[0]

    def has_converged(self, log_likelihood_closure):
        loss = -log_likelihood_closure()
        loss.backward()
        parameters = list(self.parameters())
        relative_loss_difference = math.fabs(loss.data.squeeze()[0] - self.previous_loss) / self.previous_loss

        grad_tolerance_satisfied = all([torch.norm(param.grad.data) < self._options['grad_tolerance']
                                        for param in parameters])
        loss_tolerance_satisfied = relative_loss_difference < self._options['relative_loss_tolerance']

        return grad_tolerance_satisfied or loss_tolerance_satisfied
