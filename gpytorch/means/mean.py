#!/usr/bin/env python3

from ..module import Module
from ..utils.deprecation import _ClassWithDeprecatedBatchSize


class Mean(Module, _ClassWithDeprecatedBatchSize):
    """
    Mean function.
    """
    def __init__(self):
        super().__init__()
        self._register_load_state_dict_pre_hook(self._batch_shape_state_dict_hook)

    def forward(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        # Add a last dimension
        if x.ndimension() == 1:
            x = x.unsqueeze(1)

        res = super(Mean, self).__call__(x)

        return res
