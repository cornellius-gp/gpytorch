#!/usr/bin/env python3

import math

import torch
from torch import sigmoid
from torch.nn import Module

from .. import settings
from ..utils.transforms import _get_inv_param_transform, inv_sigmoid, inv_softplus

# define softplus here instead of using torch.nn.functional.softplus because the functional version can't be pickled
softplus = torch.nn.Softplus()


class Interval(Module):
    def __init__(self, lower_bound, upper_bound, transform=sigmoid, inv_transform=inv_sigmoid, initial_value=None):
        """
        Defines an interval constraint for GP model parameters, specified by a lower bound and upper bound. For usage
        details, see the documentation for :meth:`~gpytorch.module.Module.register_constraint`.

        Args:
            lower_bound (float or torch.Tensor): The lower bound on the parameter.
            upper_bound (float or torch.Tensor): The upper bound on the parameter.
        """
        lower_bound = torch.as_tensor(lower_bound).float()
        upper_bound = torch.as_tensor(upper_bound).float()

        if torch.any(torch.ge(lower_bound, upper_bound)):
            raise RuntimeError("Got parameter bounds with empty intervals.")

        super().__init__()

        self.register_buffer("lower_bound", lower_bound)
        self.register_buffer("upper_bound", upper_bound)

        self._transform = transform
        self._inv_transform = inv_transform

        if transform is not None and inv_transform is None:
            self._inv_transform = _get_inv_param_transform(transform)

        if initial_value is not None:
            if not isinstance(initial_value, torch.Tensor):
                initial_value = torch.tensor(initial_value)
            self._initial_value = self.inverse_transform(initial_value)
        else:
            self._initial_value = None

    def _apply(self, fn):
        self.lower_bound = fn(self.lower_bound)
        self.upper_bound = fn(self.upper_bound)
        return super()._apply(fn)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        result = super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=False,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        # The lower_bound and upper_bound buffers are new, and so may not be present in older state dicts
        # Because of this, we won't have strict-mode on when loading this module
        return result

    @property
    def enforced(self):
        return self._transform is not None

    def check(self, tensor):
        return bool(torch.all(tensor <= self.upper_bound) and torch.all(tensor >= self.lower_bound))

    def check_raw(self, tensor):
        return bool(
            torch.all((self.transform(tensor) <= self.upper_bound))
            and torch.all(self.transform(tensor) >= self.lower_bound)
        )

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

        if settings.debug.on():
            max_bound = torch.max(self.upper_bound)
            min_bound = torch.min(self.lower_bound)

            if max_bound == math.inf or min_bound == -math.inf:
                raise RuntimeError(
                    "Cannot make an Interval directly with non-finite bounds. Use a derived class like "
                    "GreaterThan or LessThan instead."
                )

        transformed_tensor = (self._transform(tensor) * (self.upper_bound - self.lower_bound)) + self.lower_bound

        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        """
        Applies the inverse transformation.
        """
        if not self.enforced:
            return transformed_tensor

        if settings.debug.on():
            max_bound = torch.max(self.upper_bound)
            min_bound = torch.min(self.lower_bound)

            if max_bound == math.inf or min_bound == -math.inf:
                raise RuntimeError(
                    "Cannot make an Interval directly with non-finite bounds. Use a derived class like "
                    "GreaterThan or LessThan instead."
                )

        tensor = self._inv_transform((transformed_tensor - self.lower_bound) / (self.upper_bound - self.lower_bound))

        return tensor

    @property
    def initial_value(self):
        """
        The initial parameter value (if specified, None otherwise)
        """
        return self._initial_value

    def __repr__(self):
        if self.lower_bound.numel() == 1 and self.upper_bound.numel() == 1:
            return self._get_name() + f"({self.lower_bound:.3E}, {self.upper_bound:.3E})"
        else:
            return super().__repr__()

    def __iter__(self):
        yield self.lower_bound
        yield self.upper_bound


class GreaterThan(Interval):
    def __init__(self, lower_bound, transform=softplus, inv_transform=inv_softplus, initial_value=None):
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=math.inf,
            transform=transform,
            inv_transform=inv_transform,
            initial_value=initial_value,
        )

    def __repr__(self):
        if self.lower_bound.numel() == 1:
            return self._get_name() + f"({self.lower_bound:.3E})"
        else:
            return super().__repr__()

    def transform(self, tensor):
        transformed_tensor = self._transform(tensor) + self.lower_bound if self.enforced else tensor
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        tensor = self._inv_transform(transformed_tensor - self.lower_bound) if self.enforced else transformed_tensor
        return tensor


class Positive(GreaterThan):
    def __init__(self, transform=softplus, inv_transform=inv_softplus, initial_value=None):
        super().__init__(lower_bound=0.0, transform=transform, inv_transform=inv_transform, initial_value=initial_value)

    def __repr__(self):
        return self._get_name() + "()"

    def transform(self, tensor):
        transformed_tensor = self._transform(tensor) if self.enforced else tensor
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        tensor = self._inv_transform(transformed_tensor) if self.enforced else transformed_tensor
        return tensor


class LessThan(Interval):
    def __init__(self, upper_bound, transform=softplus, inv_transform=inv_softplus, initial_value=None):
        super().__init__(
            lower_bound=-math.inf,
            upper_bound=upper_bound,
            transform=transform,
            inv_transform=inv_transform,
            initial_value=initial_value,
        )

    def transform(self, tensor):
        transformed_tensor = -self._transform(-tensor) + self.upper_bound if self.enforced else tensor
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        tensor = -self._inv_transform(-(transformed_tensor - self.upper_bound)) if self.enforced else transformed_tensor
        return tensor

    def __repr__(self):
        return self._get_name() + f"({self.upper_bound:.3E})"
