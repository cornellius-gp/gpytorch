#!/usr/bin/env python3

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..constraints import Interval, Positive
from ..priors import Prior
from .kernel import Kernel


class ConstantKernel(Kernel):
    """
    Constant covariance kernel for the probabilistic inference of constant coefficients.

    ConstantKernel represents the prior variance `k(x1, x2) = var(c)` of a constant `c`.
    The prior variance of the constant is optimized during the GP hyper-parameter
    optimization stage. The actual value of the constant is computed (implicitly) using
    the linear algebraic approaches for the computation of GP samples and posteriors.

    The constant kernel `k_constant` is most useful as a modification of an arbitrary
    base kernel `k_base`:
    1) Additive constants: The modification `k_base + k_constant` allows the GP to
    infer a non-zero asymptotic value far from the training data, which generally
    leads to more accurate extrapolation. Notably, the uncertainty in this constant
    value affects the posterior covariances through the posterior inference equations.
    This is not the case when a constant prior mean is not used, since the prior mean
    does not show up the posterior covariance and is regularized by the log-determinant
    during the optimization of the marginal likelihood.
    2) Multiplicative constants: The modification `k_base * k_constant` allows the GP to
    modulate the variance of the kernel `k_base`, and is mathematically identical to
    `ScaleKernel(base_kernel)` with the same constant.
    """

    has_lengthscale = False

    def __init__(
        self,
        batch_shape: Optional[torch.Size] = None,
        constant_prior: Optional[Prior] = None,
        constant_constraint: Optional[Interval] = None,
        active_dims: Optional[Tuple[int, ...]] = None,
    ):
        """Constructor of ConstantKernel.

        Args:
            batch_shape: The batch shape of the kernel.
            constant_prior: Prior over the constant parameter.
            constant_constraint: Constraint to place on constant parameter.
            active_dims: The dimensions of the input with which to evaluate the kernel.
                This is mute for the constant kernel, but added for compatability with
                the Kernel API.
        """
        super().__init__(batch_shape=batch_shape, active_dims=active_dims)

        self.register_parameter(
            name="raw_constant",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )

        if constant_prior is not None:
            if not isinstance(constant_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(constant_prior).__name__)
            self.register_prior(
                "constant_prior",
                constant_prior,
                lambda m: m.constant,
                lambda m, v: m._set_constant(v),
            )

        if constant_constraint is None:
            constant_constraint = Positive()
        self.register_constraint("raw_constant", constant_constraint)

    @property
    def constant(self) -> Tensor:
        return self.raw_constant_constraint.transform(self.raw_constant)

    @constant.setter
    def constant(self, value: Tensor) -> None:
        self._set_constant(value)

    def _set_constant(self, value: Tensor) -> None:
        value = value.view(*self.batch_shape, 1)
        self.initialize(raw_constant=self.raw_constant_constraint.inverse_transform(value))

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: Optional[bool] = False,
        last_dim_is_batch: Optional[bool] = False,
    ) -> Tensor:
        """Evaluates the constant kernel.

        Args:
            x1: First input tensor of shape (batch_shape x n1 x d).
            x2: Second input tensor of shape (batch_shape x n2 x d).
            diag: If True, returns the diagonal of the covariance matrix.
            last_dim_is_batch: If True, the last dimension of size `d` of the input
                tensors are treated as a batch dimension.

        Returns:
            A (batch_shape x n1 x n2)-dim, resp. (batch_shape x n1)-dim, tensor of
            constant covariance values if diag is False, resp. True.
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        dtype = torch.promote_types(x1.dtype, x2.dtype)
        batch_shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
        shape = batch_shape + (x1.shape[-2],) + (() if diag else (x2.shape[-2],))
        constant = self.constant.to(dtype=dtype, device=x1.device)

        if not diag:
            constant = constant.unsqueeze(-1)

        if last_dim_is_batch:
            constant = constant.unsqueeze(-1)

        return constant.expand(shape)
