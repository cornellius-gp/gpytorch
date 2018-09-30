from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
