#!/usr/bin/env python3

import torch

from .mean import Mean
from ..priors import NormalPrior
from ..constraints import Positive

class QuadraticMean(Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True, weights=True):
        r"""
        Implements a mean function that is quadratic_weights x^2 + weights x + bias
        """
        super().__init__()
        self.register_parameter(name='quadratic_weights',
                                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if weights:
            self.register_parameter(name='weights',
                                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name='bias', parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))

    def forward(self, x):
        res = x.pow(2.0).matmul(self.quadratic_weights).squeeze(-1)

        if self.weights is not None:
            res = res + x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

class LogRBFMean(QuadraticMean):
    def __init__(self, input_size = 1):
        r"""
        implements the logarithm of a RBF spectral density with required constraints
        output = c - t^2 / 2l
        """
        #TODO: check that the below constraints and priors are correct
        #TODO: do these induce the proper parameterizations
        #TODO: will the priors be device independent somehow?
        
        super(LogRBFMean, self).__init__(input_size, bias=True, weights=False)
        
        self.register_prior('bias_prior', prior=NormalPrior(torch.zeros(1), 100.*torch.ones(1), transform=None), 
            param_or_closure = 'bias')
        
        self.register_constraint('quadratic_name', Positive())
        self.register_prior('quadratic_weight_prior', param_or_closure='quadratic_weights', 
            prior=NormalPrior(torch.zeros(1), 100.*torch.ones(1), transform=torch.nn.functional.softplus))


