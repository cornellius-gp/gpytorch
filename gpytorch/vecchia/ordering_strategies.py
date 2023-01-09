#!/usr/bin/env python3

import torch


# TODO: Document ordering strategies and test more orders
class OrderingStrategies:

    @staticmethod
    def coordinate_ordering(coordinate):
        return lambda data: torch.argsort(data[:, coordinate])

    @staticmethod
    def norm_ordering(*args, **kwargs):
        return lambda data: torch.argsort(torch.linalg.norm(data, *args, **kwargs))