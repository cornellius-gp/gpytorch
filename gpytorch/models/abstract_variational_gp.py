#!/usr/bin/env python3

from .gp import GP


class AbstractVariationalGP(GP):
    def __init__(self, variational_strategy):
        super(AbstractVariationalGP, self).__init__()
        self.variational_strategy = variational_strategy

    def forward(self, x):
        """
        As in the exact GP setting, the user-defined forward method should return the GP prior mean and covariance
        evaluated at input locations x.
        """
        raise NotImplementedError

    def __call__(self, inputs, **kwargs):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)

        return self.variational_strategy(inputs)
