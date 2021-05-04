#!/usr/bin/env python3

import random
from contextlib import contextmanager
from typing import Generator

import torch


def approx_equal(self, other, epsilon=1e-4):
    """
    Determines if two tensors are approximately equal
    Args:
        - self: tensor
        - other: tensor
    Returns:
        - bool
    """
    if self.size() != other.size():
        raise RuntimeError(
            "Size mismatch between self ({self}) and other ({other})".format(self=self.size(), other=other.size())
        )
    return torch.max((self - other).abs()) <= epsilon


def get_cuda_max_memory_allocations() -> int:
    """Get the `max_memory_allocated` for each cuda device"""
    return torch.tensor([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])


@contextmanager
def least_used_cuda_device() -> Generator:
    """Contextmanager for automatically selecting the cuda device
    with the least allocated memory"""
    try:
        mem_allocs = get_cuda_max_memory_allocations()
        least_used_device = torch.argmin(mem_allocs).item()
    except RuntimeError:  # raised if cuda has not been initialized
        # here we can just randomize
        least_used_device = random.randint(0, torch.cuda.device_count() - 1)
    with torch.cuda.device(least_used_device):
        yield
