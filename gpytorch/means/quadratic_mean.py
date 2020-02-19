#!/usr/bin/env python3

import torch

from .mean import Mean


class QuadraticMean(Mean):
    def __init__(self, input_size=1, batch_shape=torch.Size(), raw_weights_constraint = None,
                raw_bias_constraint = None, raw_quadratic_weights_constraint = None, weights_prior=None, bias_prior=None,
                quadratic_weights_prior=None, use_weights=True, use_bias=True,
                quadratic_weights_setting_closure=None):
        r"""
        Implements a mean function that is quadratic_weights x^2 + weights x + bias
        """
        super().__init__()
        
        self.register_parameter(name='raw_quadratic_weights',
                                parameter=torch.nn.Parameter(torch.zeros(*batch_shape, input_size, 1)))
        
        if raw_quadratic_weights_constraint:
            self.register_constraint('raw_quadratic_weights', raw_quadratic_weights_constraint)
        else:
            self.raw_quadratic_weights_constraint = None

        if use_weights:
            self.register_parameter(name='weights',
                                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))

            if raw_weights_constraint:
                self.register_constraint('raw_weights', raw_weights_constraint)
            else:
                self.raw_weights_constraint = None
        
        self.use_weights = use_weights

        if use_bias:
            self.register_parameter(name='raw_bias', parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
            
            if raw_bias_constraint:
                self.register_constraint('raw_bias', raw_bias_constraint)
            else:
                self.raw_bias_constraint = None
        
        self.use_bias = use_bias

        if quadratic_weights_prior is not None:
            if quadratic_weights_setting_closure is None:
                self.register_prior('quadratic_weights_prior', quadratic_weights_prior, 'quadratic_weights',
                                    setting_closure=quadratic_weights_setting_closure)
            else:
                self.register_prior('quadratic_weights_prior', quadratic_weights_prior, lambda: -self.quadratic_weights,
                                    setting_closure=quadratic_weights_setting_closure)                
        if weights_prior is not None:
            self.register_prior('weights_prior', weights_prior, 'weights')
        if bias_prior is not None:
            self.register_prior('bias_prior', bias_prior, 'bias')

    def forward(self, x):
        res = x.pow(2.0).matmul(self.quadratic_weights).squeeze(-1)

        if self.use_weights:
            res = res + x.matmul(self.weights).squeeze(-1)

        if self.use_bias:
            res = res + self.bias
        return res

    @property
    def quadratic_weights(self):
        if self.raw_quadratic_weights_constraint is not None:
            return self.raw_quadratic_weights_constraint.transform(self.raw_quadratic_weights)
        else:
            return self.raw_quadratic_weights

    @quadratic_weights.setter
    def quadratic_weights(self, value):
        self._set_quadratic_weights(value)
    
    def _set_quadratic_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_quadratic_weights)

        if self.raw_quadratic_weights_constraint is not None:
            self.initialize(raw_quadratic_weights=self.raw_quadratic_weights_constraint.inverse_transform(value))
        else:
            self.initialize(raw_quadratic_weights=value)

    @property
    def weights(self):
        if self.raw_weights_constraint is not None:
            return self.raw_weights_constraint.transform(self.raw_weights)
        else:
            return self.raw_weights

    @weights.setter
    def weights(self, value):
        self._set_weights(value)
    
    def _set_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weights)

        if self.raw_weights_constraint is not None:
            self.initialize(raw_weights=self.raw_weights_constraint.inverse_transform(value))
        else:
            self.initialize(raw_weights=value)

    @property
    def bias(self):
        if self.raw_bias_constraint is not None:
            return self.raw_bias_constraint.transform(self.raw_bias)
        else:
            return self.raw_bias

    @bias.setter
    def bias(self, value):
        self._set_bias(value)

    def _set_bias(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_bias)

        if self.raw_bias_constraint is not None:
            self.initialize(raw_bias=self.raw_bias_constraint.inverse_transform(value))
        else:
            self.initialize(raw_bias=value)