#!/usr/bin/env python3

import abc
import torch

### APPROACH 1 ###

class AbstractOrderingStrategy(abc.ABC):
    def __call__(self, data: torch.tensor) -> torch.LongTensor:
        output = self.order(data)
        if len(output) != len(data):
            raise Exception(
                f"Expected ordering strategy to return a tensor of length {len(data)}. "
                f"Returned a tensor of length {len(output)} instead.")
        else:
            return output

    @abc.abstractmethod
    def order(self, data: torch.tensor) -> torch.LongTensor:
        ...


class CoordinateOrdering(AbstractOrderingStrategy):
    def __init__(self, coordinate: int):
        self.coordinate = coordinate

    def order(self, data: torch.tensor) -> torch.LongTensor:
        return torch.argsort(data[:, self.coordinate]).long()


class NormOrdering(AbstractOrderingStrategy):
    def __init__(self, p: float):
        self.p = p

    def order(self, data: torch.tensor) -> torch.LongTensor:
        return torch.argsort(torch.linalg.norm(data, ord=self.p)).long()


class MSTOrdering(AbstractOrderingStrategy):
    def order(self, data: torch.tensor) -> torch.LongTensor:
        raise NotImplementedError


class MinMaxOrdering(AbstractOrderingStrategy):
    def order(self, data: torch.tensor) -> torch.LongTensor:
        raise NotImplementedError


### APPROACH 2 ###

class OrderingStrategies:
    @staticmethod
    def coordinate_ordering(coordinate: int):
        return lambda data: torch.argsort(data[:, coordinate]).long()

    @staticmethod
    def norm_ordering(p: float):
        return lambda data: torch.argsort(torch.linalg.norm(data, ord=p)).long()

    @staticmethod
    def mst_ordering():
        raise NotImplementedError

    @staticmethod
    def minmax_ordering():
        raise NotImplementedError
