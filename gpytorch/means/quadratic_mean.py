#!/usr/bin/env python3

import torch

from .mean import Mean


class QuadraticMean(Mean):
    def __init__(self, input_size=1, batch_shape=torch.Size(), weights_constraint = None,
                bias_constraint = None, quadratic_weights_constraint = None, weights_prior=None, bias_prior=None,
                quadratic_weights_prior=None):
        r"""
        Implements a mean function that is quadratic_weights x^2 + weights x + bias
        """
        super().__init__()
        
        self.register_parameter(name='quadratic_weights',
                                parameter=torch.nn.Parameter(torch.zeros(*batch_shape, input_size, 1)))
        
        self.register_parameter(name='weights',
                                parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))

        self.register_parameter(name='bias', parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))

        if quadratic_weights_constraint is not None:
            self.register_constraint('quadratic_weights', quadratic_weights_constraint)
        if weights_constraint is not None:
            self.register_constraint('weights', weights_constraint)        
        if bias_constraint is not None:
            self.register_constraint('bias', bias_constraint)

        if quadratic_weights_prior is not None:
            self.register_prior('quadratic_weights_prior', quadratic_weights_prior, 'quadratic_weights')
        if weights_prior is not None:
            self.register_prior('weights_prior', weights_prior, 'weights')
        if bias_prior is not None:
            self.register_prior('bias_prior', bias_prior, 'bias')

    def forward(self, x):
        res = x.pow(2.0).matmul(self.quadratic_weights).squeeze(-1)

        res = res + x.matmul(self.weights).squeeze(-1)

        res = res + self.bias
        return res
