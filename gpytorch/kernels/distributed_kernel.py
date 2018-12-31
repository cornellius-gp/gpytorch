#!/usr/bin/env python3

import torch
from .kernel import Kernel
from ..lazy import CatLazyTensor, NonLazyTensor
from math import ceil


class DistributedKernel(Kernel):
    r"""
    Allocates the covariance matrix on distributed devices, e.g. multiple GPUs.

    Args:
        - :attr:`base_kernel` (Kernel)
        - :attr:`device_ids` (list of `torch.device`s)
        - :attr:`output_device` `torch.device` where outputs will be placed
    """

    def __init__(self, base_kernel, device_ids, output_device=None, **kwargs):
        super(DistributedKernel, self).__init__(base_kernel=base_kernel,
                                                device_ids=device_ids,
                                                output_device=output_device,
                                                **kwargs)
        self.base_kernel = base_kernel
        self.device_ids = device_ids
        self.num_devices = len(device_ids)
        self.output_device = output_device if output_device else device_ids[0]

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return self.base_kernel.forward(x1, x2, diag=True, **params).to(self.output_device)
        # Assume x1 is `n x d` and x2 is `m x d`
        n, m = x1.size(-2), x2.size(-2)

        # Internally, x1 will always be the longer input
        # We can always compute K(x2,x1) and then transpose at the end
        if n < m:
            res = self.forward(x2, x1, **params)
            return res.transpose(-2, -1)

        x1_pieces = self._split(x1, -2, self.num_devices)

        kernel_chunks = []
        for i, x1_piece in enumerate(x1_pieces):
            chunk = self.base_kernel.forward(x1_piece, x2, **params).to(self.device_ids[i])
            if isinstance(chunk, torch.Tensor):
                chunk = NonLazyTensor(chunk)
            kernel_chunks.append(chunk)

        return CatLazyTensor(*kernel_chunks, dim=-2, output_device=self.output_device)

    def _split(self, x, dim, n_pieces):
        """
        Generator that splits `x` into `n_pieces`.
        Yields each piece of x
        """
        n = x.size(dim)
        index = [slice(None, None, None)] * x.ndimension()
        piece_len = ceil(n / n_pieces)
        for i in range(0, n, piece_len):
            index[dim] = slice(i, i + piece_len, None)
            yield x[index]
