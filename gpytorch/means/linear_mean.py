#!/usr/bin/env python3

import torch
from .mean import Mean


class LinearMean(Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True, raw_weights_constraint=None, raw_bias_constraint=None):
        super().__init__()
        self.register_parameter(name='raw_weights',
                                parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        
        if raw_weights_constraint:
            self.register_constraint('raw_weights', raw_weights_constraint)
        else:
            self.raw_weights_constraint = None

        if bias:
            self.register_parameter(name='raw_bias', parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
            
            if raw_bias_constraint:
                self.register_constraint('raw_bias', raw_bias_constraint)
            else:
                self.raw_bias_constraint = None
        else:
            self.bias = None




    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

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