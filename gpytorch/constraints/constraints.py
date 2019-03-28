#!/usr/bin/env python3

import math
import torch
from torch.nn.functional import softplus, sigmoid
from ..utils.transforms import _get_inv_param_transform
from torch.nn import Module


class Interval(Module):
    def __init__(
        self,
        lower_bound,
        upper_bound,
        transform=sigmoid,
        inv_transform=None,
    ):
        """
        Defines an interval constraint for GP model parameters, specified by a lower bound and upper bound. For usage
        details, see the documentation for :meth:`~gpytorch.module.Module.register_constraint`.

        Args:
            - lower_bound (float or torch.Tensor):
        """
        if upper_bound < math.inf and transform != sigmoid:
            raise RuntimeError("Cannot enforce an upper bound with a non-sigmoid transform!")

        super().__init__()

        if not torch.is_tensor(lower_bound):
            lower_bound = torch.tensor(lower_bound)

        if not torch.is_tensor(upper_bound):
            upper_bound = torch.tensor(upper_bound)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self._transform = transform
        if self.enforced and inv_transform is None:
            self._inv_transform = _get_inv_param_transform(transform)

    def _apply(self, fn):
        self.lower_bound = fn(self.lower_bound)
        self.upper_bound = fn(self.upper_bound)
        return super()._apply(fn)

    @property
    def enforced(self):
        return self._transform is not None

    def intersect(self, other):
        """
        Returns a new Interval constraint that is the intersection of this one and another specified one.

        Args:
            other (Interval): Interval constraint to intersect with

        Returns:
            Interval: intersection if this interval with the other one.
        """
        if self.transform != other.transform:
            raise RuntimeError("Cant intersect Interval constraints with conflicting transforms!")

        lower_bound = torch.max(self.lower_bound, other.lower_bound)
        upper_bound = torch.min(self.upper_bound, other.upper_bound)
        return Interval(lower_bound, upper_bound)

    def transform(self, tensor):
        """
        Transforms a tensor to satisfy the specified bounds.

        If upper_bound is finite, we assume that `self.transform` saturates at 1 as tensor -> infinity. Similarly,
        if lower_bound is finite, we assume that `self.transform` saturates at 0 as tensor -> -infinity.

        Example transforms for one of the bounds being finite include torch.exp and torch.nn.functional.softplus.
        An example transform for the case where both are finite is torch.nn.functional.sigmoid.
        """
        if not self.enforced:
            return tensor

        transformed_tensor = self._transform(tensor)

        upper_bound = self.upper_bound.clone()
        upper_bound[upper_bound == math.inf] = 1.
        lower_bound = self.lower_bound.clone()
        lower_bound[lower_bound == -math.inf] = 0.

        transformed_tensor = transformed_tensor * upper_bound
        transformed_tensor = transformed_tensor + lower_bound

        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        """
        Applies the inverse transformation.
        """
        if not self.enforced:
            return transformed_tensor

        upper_bound = self.upper_bound
        upper_bound[upper_bound == math.inf] = 1
        lower_bound = self.lower_bound
        lower_bound[lower_bound == -math.inf] = 0

        tensor = transformed_tensor - self.lower_bound
        tensor = tensor / self.upper_bound

        tensor = self._inv_transform(tensor)

        return tensor


class GreaterThan(Interval):
    def __init__(
        self,
        lower_bound,
        transform=softplus,
        inv_transform=None,
        active=True,
    ):
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=math.inf,
            transform=transform,
            inv_transform=inv_transform
        )


class Positive(GreaterThan):
    def __init__(self, transform=softplus, inv_transform=None):
        super().__init__(
            lower_bound=0.,
            transform=transform,
            inv_transform=inv_transform
        )


class LessThan(Interval):
    def __init__(self, upper_bound, transform=softplus, inv_transform=None):
        super().__init__(
            lower_bound=-math.inf,
            upper_bound=upper_bound,
            transform=transform,
            inv_transform=inv_transform
        )

    def transform(self, tensor):
        if not self.enforced:
            return tensor

        transformed_tensor = -self.transform(-tensor)
        transformed_tensor = transformed_tensor + self.upper_bound
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        if not self.enforced:
            return transformed_tensor

        tensor = transformed_tensor - self.upper_bound
        tensor = -self._inv_transform(-tensor)
        return tensor
