#!/usr/bin/env python3

import warnings
from .lazy_tensor import LazyTensor


class CachedSamplesLazyTensor(LazyTensor):
    def __init__(self, base_lazy_tensor, base_samples=None, num_samples=1):
        if base_samples is None:
            base_samples = base_lazy_tensor.zero_mean_mvn_samples(num_samples, samples_dim=-1)
        else:
            num_samples = base_samples.size(-1)

        super(CachedSamplesLazyTensor, self).__init__(
            base_lazy_tensor, base_samples=base_samples, num_samples=num_samples
        )
        self.base_lazy_tensor = base_lazy_tensor
        self.base_samples = base_samples
        self.num_samples = num_samples

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        return self.base_lazy_tensor._get_indices(left_indices, right_indices, *batch_indices)

    def _getitem(self, *indices):
        return self.base_lazy_tensor._getitem(*indices)

    def _matmul(self, tensor):
        return self.base_lazy_tensor._matmul(tensor)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs)

    def _size(self):
        return self.base_lazy_tensor._size()

    def _t_matmul(self, tensor):
        return self.base_lazy_tensor._t_matmul(tensor)

    def _transpose_nonbatch(self):
        return self.base_lazy_tensor._transpose_nonbatch()

    def inv_matmul(self, right_tensor, left_tensor=None):
        return self.base_lazy_tensor.inv_matmul(right_tensor, left_tensor=left_tensor)

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        return self.base_lazy_tensor.inv_quad_logdet(
            inv_quad_rhs=inv_quad_rhs, logdet=logdet, reduce_inv_quad=reduce_inv_quad
        )

    def zero_mean_mvn_samples(self, num_samples, samples_dim=0):
        if num_samples != self.num_samples:
            warnings.warn(
                "Not using the cached sampled (cached sample size is {}, asked for {} "
                "samples).".format(self.num_samples, num_samples)
            )
            return super(CachedSamplesLazyTensor, self).zero_mean_mvn_samples(num_samples, samples_dim=samples_dim)

        if samples_dim == 0:
            return self.base_samples.permute(-1, *range(self.dim() - 1)).contiguous()
        elif samples_dim == -1:
            return self.base_samples
        else:
            raise RuntimeError(
                "LazyTensor.zero_mean_mvn_samples expects samples_dim=0 or samples_dim=-1. Got {}".format(samples_dim)
            )
