#!/usr/bin/env python3

from __future__ import annotations

import math

import torch
from torch import sigmoid, Tensor
from torch.nn import Module

from ..utils.transforms import _get_inv_param_transform, inv_sigmoid, inv_softplus

# define softplus here instead of using torch.nn.functional.softplus because the functional version can't be pickled
softplus = torch.nn.Softplus()


class Interval(Module):
    def __init__(self, lower_bound, upper_bound, transform=sigmoid, inv_transform=inv_sigmoid, initial_value=None):
        """
        Defines a bounded interval constraint ``[lower_bound, upper_bound]`` for a GP model parameter.

        A constraint is enforced by a pair of inverse transformations: an unconstrained ``tensor`` is
        mapped to the constrained domain via :meth:`transform`, and a constrained value is mapped back
        via :meth:`inverse_transform`. For numerical optimization the unconstrained parameter is
        stored, and the constraint is applied on every read.

        For usage details, see the documentation for
        :meth:`~gpytorch.module.Module.register_constraint`.

        Args:
            lower_bound (float or torch.Tensor): The lower bound on the parameter.
            upper_bound (float or torch.Tensor): The upper bound on the parameter.
            transform (callable, optional): Map from the unconstrained (real) line to ``[0, 1]``.
                Defaults to ``torch.sigmoid``. Ignored for the abstract :class:`Interval` (use a
                derived class like :class:`GreaterThan` or :class:`LessThan` for one-sided bounds).
            inv_transform (callable, optional): Inverse of ``transform``. If ``None``, an inverse is
                derived automatically from the registered transforms module.
            initial_value (float or torch.Tensor, optional): A value within the interval used to
                initialize the underlying unconstrained parameter when the constraint is registered.

        Example:
            >>> import torch
            >>> from gpytorch.constraints import Interval
            >>> # Constrain a parameter to live in [0.1, 1.0]
            >>> constraint = Interval(0.1, 1.0)
            >>> raw = torch.tensor(0.0)  # unconstrained
            >>> constraint.transform(raw)
            tensor(0.5500)
            >>> constraint.inverse_transform(constraint.transform(raw)).item()
            0.0
        """
        dtype = torch.get_default_dtype()
        lower_bound = torch.as_tensor(lower_bound).to(dtype)
        upper_bound = torch.as_tensor(upper_bound).to(dtype)

        if torch.any(torch.ge(lower_bound, upper_bound)):
            raise ValueError("Got parameter bounds with empty intervals.")

        if type(self) is Interval:
            max_bound = torch.max(upper_bound)
            min_bound = torch.min(lower_bound)
            if max_bound == math.inf or min_bound == -math.inf:
                raise ValueError(
                    "Cannot make an Interval directly with non-finite bounds. Use a derived class like "
                    "GreaterThan or LessThan instead."
                )

        super().__init__()

        self.register_buffer("lower_bound", lower_bound)
        self.register_buffer("upper_bound", upper_bound)

        self._transform = transform
        self._inv_transform = inv_transform

        if transform is not None and inv_transform is None:
            self._inv_transform = _get_inv_param_transform(transform)

        if initial_value is not None:
            self._initial_value = self.inverse_transform(torch.as_tensor(initial_value))
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
    def enforced(self) -> bool:
        return self._transform is not None

    def check(self, tensor) -> bool:
        return bool(torch.all(tensor <= self.upper_bound) and torch.all(tensor >= self.lower_bound))

    def check_raw(self, tensor) -> bool:
        return bool(
            torch.all(self.transform(tensor) <= self.upper_bound)
            and torch.all(self.transform(tensor) >= self.lower_bound)
        )

    def intersect(self, other: Interval) -> Interval:
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

    def transform(self, tensor: Tensor) -> Tensor:
        """
        Transforms a tensor to satisfy the specified bounds.

        If upper_bound is finite, we assume that ``self._transform`` saturates at 1 as tensor -> infinity.
        Similarly, if lower_bound is finite, we assume that ``self._transform`` saturates at 0 as
        tensor -> -infinity.

        Example transforms for one of the bounds being finite include ``torch.exp`` and
        ``torch.nn.functional.softplus``. An example transform for the case where both are finite is
        ``torch.nn.functional.sigmoid``.

        Args:
            tensor (torch.Tensor): The unconstrained tensor to be transformed.

        Returns:
            torch.Tensor: A tensor whose values lie in ``[lower_bound, upper_bound]``.
        """
        if not self.enforced:
            return tensor

        transformed_tensor = (self._transform(tensor) * (self.upper_bound - self.lower_bound)) + self.lower_bound

        return transformed_tensor

    def inverse_transform(self, transformed_tensor: Tensor) -> Tensor:
        """
        Applies the inverse transformation, mapping a constrained tensor back to its unconstrained
        representation.

        Args:
            transformed_tensor (torch.Tensor): A tensor whose values lie in
                ``[lower_bound, upper_bound]``.

        Returns:
            torch.Tensor: The unconstrained tensor.
        """
        if not self.enforced:
            return transformed_tensor

        tensor = self._inv_transform((transformed_tensor - self.lower_bound) / (self.upper_bound - self.lower_bound))

        return tensor

    @property
    def initial_value(self) -> Tensor | None:
        """
        The initial value assigned to the constrained parameter at registration time
        (if one was provided to :meth:`__init__`, otherwise ``None``).
        """
        return self._initial_value

    def __repr__(self) -> str:
        if self.lower_bound.numel() == 1 and self.upper_bound.numel() == 1:
            return self._get_name() + f"({self.lower_bound:.3E}, {self.upper_bound:.3E})"
        else:
            return super().__repr__()

    def __iter__(self):
        yield self.lower_bound
        yield self.upper_bound


class GreaterThan(Interval):
    def __init__(self, lower_bound, transform=softplus, inv_transform=inv_softplus, initial_value=None):
        """
        Defines a lower-bound constraint ``x >= lower_bound`` for a GP model parameter.

        Uses the softplus transform by default, which is the natural choice for strictly positive
        quantities and is the most common constraint in GPyTorch (e.g. lengthscales, noise scales,
        outputscales).

        Args:
            lower_bound (float or torch.Tensor): The lower bound on the parameter.
            transform (callable, optional): Map from the unconstrained (real) line to ``[0, infinity)``.
                Defaults to ``torch.nn.functional.softplus``.
            inv_transform (callable, optional): Inverse of ``transform``. Defaults to the inverse
                softplus.
            initial_value (float or torch.Tensor, optional): A value ``>= lower_bound`` used to
                initialize the underlying unconstrained parameter when the constraint is registered.

        Example:
            >>> from gpytorch.constraints import GreaterThan
            >>> constraint = GreaterThan(1e-3)  # typical lengthscale lower bound
        """
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=math.inf,
            transform=transform,
            inv_transform=inv_transform,
            initial_value=initial_value,
        )

    def __repr__(self) -> str:
        if self.lower_bound.numel() == 1:
            return self._get_name() + f"({self.lower_bound:.3E})"
        else:
            return super().__repr__()

    def transform(self, tensor: Tensor) -> Tensor:
        transformed_tensor = self._transform(tensor) + self.lower_bound if self.enforced else tensor
        return transformed_tensor

    def inverse_transform(self, transformed_tensor: Tensor) -> Tensor:
        tensor = self._inv_transform(transformed_tensor - self.lower_bound) if self.enforced else transformed_tensor
        return tensor


class Positive(GreaterThan):
    def __init__(self, transform=softplus, inv_transform=inv_softplus, initial_value=None):
        """
        Defines a positivity constraint ``x > 0`` for a GP model parameter.

        This is a special case of :class:`GreaterThan` with ``lower_bound=0`` and is the most
        commonly used constraint in GPyTorch.

        Args:
            transform (callable, optional): Map from the unconstrained (real) line to ``[0, infinity)``.
                Defaults to ``torch.nn.functional.softplus``.
            inv_transform (callable, optional): Inverse of ``transform``. Defaults to the inverse
                softplus.
            initial_value (float or torch.Tensor, optional): A strictly positive value used to
                initialize the underlying unconstrained parameter when the constraint is registered.

        Example:
            >>> from gpytorch.constraints import Positive
            >>> constraint = Positive()  # equivalent to GreaterThan(0.0)
        """
        super().__init__(lower_bound=0.0, transform=transform, inv_transform=inv_transform, initial_value=initial_value)

    def __repr__(self) -> str:
        return self._get_name() + "()"

    def transform(self, tensor: Tensor) -> Tensor:
        transformed_tensor = self._transform(tensor) if self.enforced else tensor
        return transformed_tensor

    def inverse_transform(self, transformed_tensor: Tensor) -> Tensor:
        tensor = self._inv_transform(transformed_tensor) if self.enforced else transformed_tensor
        return tensor


class LessThan(Interval):
    def __init__(self, upper_bound, transform=softplus, inv_transform=inv_softplus, initial_value=None):
        """
        Defines an upper-bound constraint ``x <= upper_bound`` for a GP model parameter.

        Uses the softplus transform on the negated tensor, which is the natural choice for strictly
        bounded above quantities.

        Args:
            upper_bound (float or torch.Tensor): The upper bound on the parameter.
            transform (callable, optional): Map from the unconstrained (real) line to ``[0, infinity)``.
                Defaults to ``torch.nn.functional.softplus``.
            inv_transform (callable, optional): Inverse of ``transform``. Defaults to the inverse
                softplus.
            initial_value (float or torch.Tensor, optional): A value ``<= upper_bound`` used to
                initialize the underlying unconstrained parameter when the constraint is registered.

        Example:
            >>> from gpytorch.constraints import LessThan
            >>> constraint = LessThan(1.0)  # parameter must stay <= 1.0
        """
        super().__init__(
            lower_bound=-math.inf,
            upper_bound=upper_bound,
            transform=transform,
            inv_transform=inv_transform,
            initial_value=initial_value,
        )

    def transform(self, tensor: Tensor) -> Tensor:
        transformed_tensor = -self._transform(-tensor) + self.upper_bound if self.enforced else tensor
        return transformed_tensor

    def inverse_transform(self, transformed_tensor: Tensor) -> Tensor:
        tensor = -self._inv_transform(-(transformed_tensor - self.upper_bound)) if self.enforced else transformed_tensor
        return tensor

    def __repr__(self) -> str:
        return self._get_name() + f"({self.upper_bound:.3E})"
