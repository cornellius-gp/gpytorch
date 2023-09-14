#!/usr/bin/env python3

import torch


class OrderingStrategies:
    @staticmethod
    def coordinate_ordering(coordinate: int):
        return lambda data: torch.argsort(data[:, coordinate]).long()

    @staticmethod
    def norm_ordering(p: float, dim: int):
        return lambda data: torch.argsort(torch.linalg.norm(data, ord=p, dim=dim)).long()

    @staticmethod
    def mst_ordering():
        raise NotImplementedError

    @staticmethod
    def minmax_ordering():
        raise NotImplementedError
