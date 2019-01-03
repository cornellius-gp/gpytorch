#!/usr/bin/env python3

import torch
from torch.nn.parallel import DataParallel
from .kernel import Kernel
from ..lazy import CatLazyTensor, lazify
from .. import settings


class MultiDeviceKernel(DataParallel, Kernel):
    r"""
    Allocates the covariance matrix on distributed devices, e.g. multiple GPUs.

    Args:
        - :attr:`base_kernel`: Base kernel to distribute
        - :attr:`device_ids`: list of `torch.device` objects to place kernel chunks on
        - :attr:`output_device`: Device where outputs will be placed
    """

    def __init__(self, base_kernel, device_ids, output_device=None, **kwargs):
        DataParallel.__init__(self,
                              module=base_kernel,
                              device_ids=device_ids,
                              output_device=output_device,
                              dim=-2)

        self.output_device = output_device if output_device else device_ids[0]

        self.__cached_x1 = torch.empty(1)
        self.__cached_x2 = torch.empty(1)

    def forward(self, x1, x2, diag=False, **kwargs):
        if diag:
            return self.module.forward(x1, x2, diag=True, **kwargs).to(self.output_device)

        if not x1.device == self.__cached_x1.device or not torch.equal(x1, self.__cached_x1):
            self._x1_scattered, self._kwargs = self.scatter((x1,), kwargs, self.device_ids)
            self.__cached_x1 = x1

        if not x2.device == self.__cached_x2.device or not torch.equal(x2, self.__cached_x2):
            self._x2_subs = [x2.to(x1_[0].device) for x1_ in self._x1_scattered]
            self.__cached_x2 = x2

        inputs = tuple((x1_[0], x2_) for x1_, x2_ in zip(self._x1_scattered, self._x2_subs))

        if not self.device_ids:
            return self.module.forward(*inputs, **self._kwargs)

        if len(self.device_ids) == 1:
            return self.module.forward(*inputs[0], **self._kwargs[0])

        # Can't cache the replication because the base kernel module can change every time (e.g. param updates)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

        # TODO: parallel_apply might be too heavyweight in some cases?
        with settings.lazily_evaluate_kernels(False):
            outputs = self.parallel_apply(replicas, inputs, self._kwargs)

        return self.gather(outputs, self.output_device)

    def gather(self, outputs, output_device):
        return CatLazyTensor(*[lazify(o) for o in outputs], dim=self.dim, output_device=self.output_device)
