#!/usr/bin/env python3

import torch
from .sum_lazy_tensor import SumLazyTensor


class PsdSumLazyTensor(SumLazyTensor):
    """
    A SumLazyTensor, but where every component of the sum is positive semi-definite
    """

    def zero_mean_mvn_samples(self, sample_shape=torch.Size()):
        return sum(lazy_tensor.zero_mean_mvn_samples(sample_shape=sample_shape) for lazy_tensor in self.lazy_tensors)
