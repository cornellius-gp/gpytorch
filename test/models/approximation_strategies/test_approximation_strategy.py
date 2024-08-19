from __future__ import annotations

import abc
import unittest
from typing import Tuple

import torch
from jaxtyping import Float


class ApproximationStrategyTestCase(abc.ABC, unittest.TestCase):
    @abc.abstractmethod
    def create_model(self, train_x, train_y, likelihood):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_likelihood(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_test_data(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_train_data(self) -> Tuple[Float[torch.Tensor, "N D"], Float[torch.Tensor, " N"]]:
        raise NotImplementedError()
