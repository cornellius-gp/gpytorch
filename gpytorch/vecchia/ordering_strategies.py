#!/usr/bin/env python3

import torch
from ._blocker import BaseBlocker


# TODO: Document ordering strategies and test more orders
class OrderingStrategies:

    @staticmethod
    def coordinate_ordering(coordinate):
        return lambda data: torch.argsort(data[:, coordinate]).long()

    @staticmethod
    def norm_ordering(*args, **kwargs):
        return lambda data: torch.argsort(torch.linalg.norm(data, *args, **kwargs)).long()

    @staticmethod
    def minimax_ordering(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def mst_ordering(*args, **kwargs):
        raise NotImplementedError
