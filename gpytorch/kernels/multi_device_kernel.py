#!/usr/bin/env python3

import torch
from torch.nn.parallel import DataParallel
from .kernel import Kernel
from ..lazy import CatLazyTensor, NonLazyTensor
from math import ceil


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

    def forward(self, x1, x2, diag=False, **kwargs):
        if diag:
            return self.module.forward(x1, x2, diag=True, **kwargs).to(self.output_device)
         
        x1_scattered, kwargs = self.scatter((x1,), kwargs, self.device_ids)
        inputs = tuple((x1_[0], x2.to(x1_[0].device)) for x1_ in x1_scattered)

        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def gather(self, outputs, output_device):
        return CatLazyTensor(*outputs, dim=self.dim, output_device=self.output_device)
