#!/usr/bin/env python3

import abc
import torch

### APPROACH 1 ###

class AbstractDistanceMetric(abc.ABC):
    @abc.abstractmethod
    def measure(self, x1: torch.tensor, x2: torch.tensor) -> torch.FloatTensor:
        ...

    def __call__(self, x1: torch.tensor, x2: torch.tensor) -> torch.FloatTensor:
        output = self.measure(x1, x2)
        if output.shape != torch.Size([x1.shape[0], x2.shape[0]]):
            raise Exception(
                f"Expected distance metric to return an {(x1.shape[0], x2.shape[0])} tensor given inputs of shapes "
                f"{tuple(x1.shape)} and {tuple(x2.shape)}. Returned an {tuple(output.shape)} tensor instead.")
        else:
            return output


class EuclideanDistance(AbstractDistanceMetric):
    def measure(self, x1: torch.tensor, x2: torch.tensor) -> torch.FloatTensor:
        return torch.cdist(x1, x2, p=2).float()


class ManhattanDistance(AbstractDistanceMetric):
    def measure(self, x1: torch.tensor, x2: torch.tensor) -> torch.FloatTensor:
        return torch.cdist(x1, x2, p=1).float()


### APPROACH 2 ###

class DistanceMetrics:

    @staticmethod
    def euclidean_distance():
        return lambda x1, x2: torch.cdist(x1, x2, p=2).float()

    @staticmethod
    def manhattan_distance():
        return lambda x1, x2: torch.cdist(x1, x2, p=1).float()

    @staticmethod
    def mst_distance():
        raise NotImplementedError
