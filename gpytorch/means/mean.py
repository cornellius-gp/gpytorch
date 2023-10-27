#!/usr/bin/env python3

from typing import Iterable, Union

from torch import Tensor

from torch.nn import ModuleList

from ..module import Module


class Mean(Module):
    """
    Mean function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        # Add a last dimension
        if x.ndimension() == 1:
            x = x.unsqueeze(1)

        res = super(Mean, self).__call__(x)

        return res

    def __add__(self, other: "Mean") -> "Mean":
        means = []
        for entity in [self, other]:
            means += entity.means if isinstance(entity, AdditiveMean) else [entity]
        return AdditiveMean(*means)

    def __sub__(self, other: "Mean") -> "Mean":
        means = []
        negative_means = []
        means += self.means if isinstance(self, AdditiveMean) else [self]
        negative_means += other.means if isinstance(other, AdditiveMean) else [other]
        return AdditiveMean(*means, negative_means=negative_means)


class AdditiveMean(Mean):
    """
    A Mean that supports summing over multiple component means.

    Example:
        >>> mean_module = LinearMean(2) + PositiveQuadraticMean(2)
        >>> x1 = torch.randn(50, 2)
        >>> additive_mean_vector = mean_module(x1)

    :param means: Means to add together.
    :param negative_means: Means to subtract from the sum.
    """

    def __init__(self, *means: Iterable[Mean], negative_means: Union[Iterable[Mean], None] = None):
        super(AdditiveMean, self).__init__()
        self.means = ModuleList(means)
        self.negative_means = ModuleList(negative_means) if negative_means is not None else None

    def forward(self, x: Tensor) -> Tensor:
        res = 0.0
        for mean in self.means:
            res += mean(x)
        if self.negative_means is not None:
            for mean in self.negative_means:
                res -= mean(x)
        return res
