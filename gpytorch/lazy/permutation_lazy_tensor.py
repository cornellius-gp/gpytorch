#!/usr/bin/env python3

import torch
from .lazy_tensor import LazyTensor


class PermutationLazyTensor(LazyTensor):
    def __init__(self, pi):
        super().__init__(pi)

        self.pi = pi

    def _size(self):
        return self.pi.size(-1), self.pi.size(-1)

    def _transpose_nonbatch(self):
        return PermutationLazyTensor(self.pi.argsort())

    def inv_matmul(self, rhs):
        return self.transpose(-2, -1).matmul(rhs)

    def diag(self):
        assert 0

    @property
    def dtype(self):
        """
        Maybe add a dtype argument in self.__init__()?
        """
        return torch.float32

    def logdet(self):
        """
        Technically, det(P) could be either 1 or -1.
        Thus, logdet(P) could be NaN.
        But returning 0 is fine if P and P' always appear in pairs.
        """
        return 0.

    def _matmul(self, rhs):
        return rhs[..., self.pi, :]
