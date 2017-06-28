import torch
from .parameter_group import ParameterGroup
from torch.nn import Parameter
from torch.autograd import Variable
from ..utils import pd_catcher, LBFGS

class MLEParameterGroup(ParameterGroup):
    def __init__(self, **kwargs):
        for name, param in kwargs.items():
            if not isinstance(param, Parameter):
                raise RuntimeError('All parameters in an MLEParameterGroup must be Parameters')
            setattr(self, name, param)
        self._update_options = {}


    def update(self, log_likelihood_closure):
        _, parameters = zip(*self)
        optimizer = LBFGS(parameters, line_search_fn='backtracking', **self._update_options)
        optimizer.n_iter = 0

        @pd_catcher(catch_function=lambda: Variable(torch.Tensor([10000])))
        def step_closure():
            optimizer.zero_grad()
            optimizer.n_iter += 1
            loss = -log_likelihood_closure()
            loss.backward()
            return loss

        optimizer.step(step_closure)
