#!/usr/bin/env python3

import torch


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
