#!/usr/bin/env python3

from ..module import Module


class Mean(Module):
    """
    """

    def forward(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        # Add a last dimension
        if x.ndimension() == 1:
            x = x.unsqueeze(1)

        is_batch = x.ndimension() == 3

        if not is_batch:
            x = x.unsqueeze(0)

        res = super(Mean, self).__call__(x)

        if not is_batch:
            res = res[0]

        return res
