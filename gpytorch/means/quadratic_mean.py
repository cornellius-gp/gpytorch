#!/usr/bin/env python3

import torch

from .mean import Mean
#from ..priors import NormalPrior
#from ..constraints import LessThan


class QuadraticMean(Mean):
    def __init__(self, input_size=1, batch_shape=torch.Size(), bias=True, weights=True):
        r"""
        Implements a mean function that is quadratic_weights x^2 + weights x + bias
        """
        super().__init__()
        self.register_parameter(name='quadratic_weights',
                                parameter=torch.nn.Parameter(torch.zeros(*batch_shape, input_size, 1)))
        if weights:
            self.register_parameter(name='weights',
                                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        else:
            self.weights = None

        if bias:
            self.register_parameter(name='bias', parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        res = x.pow(2.0).matmul(self.quadratic_weights).squeeze(-1)

        if self.weights is not None:
            res = res + x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

    # @property
    # def quadratic_weights(self):
    #     return self.quadratic_weights

    # @property
    # def weights(self):
    #     return self.weights
    
    # @property
    # def bias(self):
    #     return self.bias

    # @quadratic_weights.setter
    # def quadratic_weights(self, value):
    #     self._set_quadratic_weights(value)
    
    # def _set_quadratic_weights(self, value):
    #     if not torch.is_tensor(value):
    #         value = torch.as_tensor(value).to(self.quadratic_weights)
    #     self.initialize(quadratic_weights=value)

    # @weights.setter
    # def weights(self, value):
    #     self._set_weights(value)
    
    # def _weights(self, value):
    #     if not torch.is_tensor(value):
    #         value = torch.as_tensor(value).to(self.weights)
    #     self.initialize(weights=value)

    # @bias.setter
    # def bias(self, value):
    #     self._set_bias(value)
    
    # def _set_bias(self, value):
    #     if not torch.is_tensor(value):
    #         value = torch.as_tensor(value).to(self.bias)
    #     self.initialize(bias=value)

