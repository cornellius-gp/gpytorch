#!/usr/bin/env python3

from ..module import Module


class Mean(Module):
    """
    Mean function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        # Add a last dimension
        if x.ndimension() == 1:
            x = x.unsqueeze(1)

        res = super(Mean, self).__call__(x)

        return res
