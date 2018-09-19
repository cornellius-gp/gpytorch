from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import torch
from .random_variable import RandomVariable


class SamplesRandomVariable(RandomVariable):
    def __init__(self, samples):
        """
        Constructs a random variable from samples

        samples should be a Tensor, with the first dimension representing the samples

        Params:
        - samples (Tensor: b x ...) samples
        """
        if not torch.is_tensor(samples):
            raise RuntimeError("samples should be a Tensor")
        super(SamplesRandomVariable, self).__init__(samples)
        self._samples = samples

    def sample(self, n_samples=None, **kwargs):
        squeeze = False
        if n_samples is None:
            n_samples = 1
            squeeze = False

        ix = random.randrange(len(self._samples))[:n_samples]
        res = self._samples[ix]

        if squeeze:
            res = res.squeeze(0)
        return res

    def representation(self):
        return self._samples

    def mean(self):
        return self._samples.mean(0)

    def var(self):
        if self._samples.size(0) == 1:
            return torch.zeros_like(self._samples.squeeze(0))
        return self._samples.var(0)
