#!/usr/bin/env python3
import torch

from . import lazify, KroneckerProductLazyTensor

class DiagKronProdLazyTensor(KroneckerProductLazyTensor):
    def __init__(self, *lazy_tensors):
        try:
            lazy_tensors = tuple(lazify(lazy_tensor) for lazy_tensor in lazy_tensors)
            # print(lazy_tensors)
        except TypeError:
            raise RuntimeError("DiagKronProductLazyTensor is intended to wrap lazy tensors.")
        for prev_lazy_tensor, curr_lazy_tensor in zip(lazy_tensors[:-1], lazy_tensors[1:]):
            if prev_lazy_tensor.batch_shape != curr_lazy_tensor.batch_shape:
                raise RuntimeError(
                    "DiagKronProductLazyTensor expects lazy tensors with the "
                    "same batch shapes. Got {}.".format([lv.batch_shape for lv in lazy_tensors])
                )
        super(DiagKronProdLazyTensor, self).__init__(*lazy_tensors)
        self.lazy_tensors = lazy_tensors

    def get_diag(self):
        sz1 = self.lazy_tensors[0].size(0)
        sz2 = self.lazy_tensors[1].size(0)

        d1 = self.lazy_tensors[0].diag()
        d2 = self.lazy_tensors[1].diag()

        out = d2.expand(sz1, sz2).t().mul(d1)

        return out.t().contiguous().view(sz1*sz2)