import math
import torch
from .parameter_group import ParameterGroup
from torch.autograd import Variable
from ..utils import pd_catcher, LBFGS
from .bounded_parameter import BoundedParameter

import pdb

class MLEParameterGroup(ParameterGroup):
    def __init__(self, **kwargs):
        super(MLEParameterGroup, self).__init__()
        self._bounds = {}

        for name, param in kwargs.items():
            if not isinstance(param, BoundedParameter):
                raise RuntimeError('All parameters in an MLEParameterGroup must be BoundedParameters')
            setattr(self, name, param.param)
            self._bounds[name] = torch.Tensor([param.lower_bound, param.upper_bound]).unsqueeze_(1).t_()
        self._options = {
            'grad_tolerance': 1e-5,
            'relative_loss_tolerance': 1e-3,
            'optim_options': {
                'max_iter': 50
            },
        }

    def __repr__(self):
        string = 'MLEParameterGroup(\n'
        for name, param in self.named_parameters():
            string += '\t - ' + name + ' - ' + str(param.data) + '\n'
        string += ')'
        return string

    def update(self, log_likelihood_closure):
        parameters = list(self.parameters())
        bounds = torch.cat([self._bounds[name] for name, _ in self.named_parameters()])
        optim_options = self._options['optim_options']
        optimizer = LBFGS(parameters, line_search_fn='goldstein', lr=0.001, bounds=bounds, **optim_options)
        optimizer.n_iter = 0

        @pd_catcher(catch_function=lambda: Variable(torch.Tensor([10000])))
        def step_closure():
            optimizer.zero_grad()
            optimizer.n_iter += 1
            loss = -log_likelihood_closure()
            loss.backward()
            return loss

        loss = optimizer.step(step_closure)
        if isinstance(loss, Variable):
            self.previous_loss = loss.data.squeeze()[0]
        else:
            self.previous_loss = loss

    def has_converged(self, log_likelihood_closure):
        loss = -log_likelihood_closure()
        loss.backward()
        parameters = list(self.parameters())
        relative_loss_difference = math.fabs(loss.data.squeeze()[0] - self.previous_loss) / self.previous_loss

        grad_tolerance_satisfied = all([torch.norm(param.grad.data) < self._options['grad_tolerance']
                                        for param in parameters])
        loss_tolerance_satisfied = relative_loss_difference < self._options['relative_loss_tolerance']

        return grad_tolerance_satisfied or loss_tolerance_satisfied
