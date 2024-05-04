#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from ..functions import MaternCovariance
from ..settings import trace_mode
from ..typing import Float
from .kernel import Kernel


class MaternKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Matern kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{Matern}}(\mathbf{x_1}, \mathbf{x_2}) = \frac{2^{1 - \nu}}{\Gamma(\nu)}
          \left( \sqrt{2 \nu} d \right)^{\nu} K_\nu \left( \sqrt{2 \nu} d \right)
       \end{equation*}

    where

    * :math:`d = (\mathbf{x_1} - \mathbf{x_2})^\top \Theta^{-2} (\mathbf{x_1} - \mathbf{x_2})`
      is the distance between
      :math:`x_1` and :math:`x_2` scaled by the lengthscale parameter :math:`\Theta`.
    * :math:`\nu` is a smoothness parameter (takes values 1/2, 3/2, or 5/2). Smaller values are less smooth.
    * :math:`K_\nu` is a modified Bessel function.

    There are a few options for the lengthscale parameter :math:`\Theta`:
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    :param nu: (Default: 2.5) The smoothness parameter.
    :type nu: float (0.5, 1.5, or 2.5)
    :param ard_num_dims: (Default: `None`) Set this if you want a separate lengthscale for each
        input dimension. It should be `d` if x1 is a `... x n x d` matrix.
    :type ard_num_dims: int, optional
    :param batch_shape: (Default: `None`) Set this if you want a separate lengthscale for each
         batch of input data. It should be `torch.Size([b1, b2])` for a `b1 x b2 x n x m` kernel output.
    :type batch_shape: torch.Size, optional
    :param active_dims: (Default: `None`) Set this if you want to
        compute the covariance of only a few input dimensions. The ints
        corresponds to the indices of the dimensions.
    :type active_dims: Tuple(int)
    :param lengthscale_prior: (Default: `None`)
        Set this if you want to apply a prior to the lengthscale parameter.
    :type lengthscale_prior: ~gpytorch.priors.Prior, optional
    :param lengthscale_constraint: (Default: `Positive`) Set this if you want
        to apply a constraint to the lengthscale parameter.
    :type lengthscale_constraint: ~gpytorch.constraints.Interval, optional
    :param eps: (Default: 1e-6) The minimum value that the lengthscale can take (prevents divide by zero errors).
    :type eps: float, optional

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=5))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.MaternKernel(nu=0.5, batch_shape=torch.Size([2])
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)
    """

    has_lengthscale = True

    def __init__(self, nu: Optional[float] = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternKernel, self).__init__(**kwargs)
        self.nu = nu

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            mean = x1.mean(dim=-2, keepdim=True)

            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
            return constant_component * exp_component
        return MaternCovariance.apply(
            x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )

    def _forward_helper(
        self, X1: Float[Tensor, "batch* M D"], X2: Float[Tensor, "batch* N D"]  # noqa F722
    ) -> Float[Tensor, "batch* M N"]:  # noqa F722
        X1_ = X1[..., :, None, :]
        X2_ = X2[..., None, :, :]
        diffs = X1_ - X2_
        if diffs.shape[-1] > 1:  # No special casing here causes 10x slowdown!
            dists = diffs.norm(dim=-1)
        else:
            dists = diffs.abs().squeeze(-1)
        consts = -math.sqrt(self.nu * 2)
        exp_component = (consts * dists).exp_()

        if self.nu == 0.5:
            constant_component = torch.ones_like(dists)
        elif self.nu == 1.5:
            constant_component = torch.add(1.0, dists, alpha=math.sqrt(3))
        elif self.nu == 2.5:
            constant_component = torch.add(1.0, dists, alpha=math.sqrt(5)).add_(dists.square(), alpha=(5.0 / 3.0))

        return diffs, dists, exp_component, constant_component

    def _forward(
        self, X1: Float[Tensor, "batch* M D"], X2: Float[Tensor, "batch* N D"]  # noqa F722
    ) -> Float[Tensor, "batch* M N"]:  # noqa F722
        r"""Evaluate the Matern kernel.

        O(NMD) time
        O(NMD) memory

        :param X1: Kernel input :math:`\boldsymbol X_1`
        :param X2: Kernel input :math:`\boldsymbol X_2`
        :return: The Matern kernel matrix :math:`\boldsymbol K(\boldsymbol X_1, \boldsymbol X_2)`

        .. note::

            This function does not broadcast. `X1` and `X2` must have the same batch shapes.
        """
        _, _, exp_component, constant_component = self._forward_helper(X1, X2)
        return exp_component.mul_(constant_component)

    def _forward_no_kernel_linop(
        self, X1: Float[Tensor, "batch* M D"], X2: Float[Tensor, "batch* N D"]  # noqa F722
    ) -> Float[Tensor, "batch* M N"]:  # noqa F722
        X1_ = X1[..., :, None, :]
        X2_ = X2[..., None, :, :]
        diffs = X1_ - X2_
        if diffs.shape[-1] > 1:  # No special casing here causes 10x slowdown!
            dists = diffs.norm(dim=-1)
        else:
            dists = diffs.abs().squeeze(-1)
        consts = -math.sqrt(self.nu * 2)
        exp_component = (consts * dists).exp()

        if self.nu == 0.5:
            constant_component = torch.ones_like(dists)
        elif self.nu == 1.5:
            constant_component = torch.add(1.0, dists, alpha=math.sqrt(3))
        elif self.nu == 2.5:
            constant_component = torch.add(1.0, dists, alpha=math.sqrt(5)).add_(dists.square(), alpha=(5.0 / 3.0))

        return exp_component.mul(constant_component)

    def _vjp(
        self,
        V: Float[Tensor, "*batch M N"],  # noqa F722
        X1: Float[Tensor, "*batch M D"],  # noqa F722
        X2: Float[Tensor, "*batch N D"],  # noqa F722
    ) -> Tuple[Float[Tensor, "*batch M D"], Float[Tensor, "*batch N D"]]:  # noqa F722
        diffs, dists, exp_component, constant_component = self._forward_helper(X1=X1, X2=X2)

        if self.nu == 0.5:
            d_constant_component = torch.tensor(0.0, dtype=X1.dtype, device=X1.device)
        elif self.nu == 1.5:
            d_constant_component = torch.tensor(math.sqrt(3), dtype=X1.dtype, device=X1.device)
        elif self.nu == 2.5:
            d_constant_component = torch.add(math.sqrt(5), dists, alpha=(10.0 / 3.0))

        # Product rule:
        # dK_ddists = (d_constant_component * exp_component) + (constant_component * d_exp_component)
        # d_exp_component = consts * exp_component
        V_dK_ddists = (
            torch.add(d_constant_component, constant_component, alpha=-math.sqrt(self.nu * 2))
            .mul_(exp_component)
            .mul_(V)
        )
        # dK_dX1 = dK_ddists * ddists_dX1
        # ddists_dX1 = X1 / ||X1||
        V_dK_ddists_div_dists = V_dK_ddists.div_(dists).nan_to_num_(nan=0.0)
        X1_grad = torch.einsum("...MN,...MND->...MD", V_dK_ddists_div_dists, diffs)
        X2_grad = torch.einsum("...MN,...MND->...ND", V_dK_ddists_div_dists, diffs).mul_(-1)
        return X1_grad, X2_grad

    def _forward_and_vjp(
        self,
        X1: Float[Tensor, "*batch M D"],
        X2: Float[Tensor, "*batch N D"],
        V: Optional[Float[Tensor, "*batch M N"]] = None,
    ) -> Tuple[Float[Tensor, "*batch M N"], Tuple[Float[Tensor, "*batch M D"], Float[Tensor, "*batch N D"]]]:
        r"""
        O(NMD) time
        O(NMD) memory

        :param X1: Kernel input :math:`\boldsymbol X_1`
        :param X2: Kernel input :math:`\boldsymbol X_2`
        :param V: :math:`\boldsymbol V` - the LHS of the VJP operation
        :return: The kernel matrix :math:`\boldsymbol K` and a tuple containing the VJPs
            :math:`\frac{\del \mathrm{tr} \left( \boldsymbol V^\top \boldsymbol K(\boldsymbol X_1, \boldsymbol X_2) \right)}{\del \boldsymbol X_1}`
            and
            :math:`\frac{\del \mathrm{tr} \left( \boldsymbol V^\top \boldsymbol K(\boldsymbol X_1, \boldsymbol X_2) \right)}{\del \boldsymbol X_2}`

        .. note::

            This function does not broadcast. `V`, `X1`, and `X2` must have the same batch shapes.
        """  # noqa: E501
        # Compute quantities used in forward and backward pass
        diffs, dists, exp_component, constant_component = self._forward_helper(X1=X1, X2=X2)

        # Forward pass K(X1, X2)
        K = exp_component * constant_component

        # Backward pass / Vector-Jacobian Product
        if self.nu == 0.5:
            d_constant_component = torch.tensor(0.0, dtype=X1.dtype, device=X1.device)
        elif self.nu == 1.5:
            d_constant_component = torch.tensor(math.sqrt(3), dtype=X1.dtype, device=X1.device)
        elif self.nu == 2.5:
            d_constant_component = torch.add(math.sqrt(5), dists, alpha=(10.0 / 3.0))

        # Product rule:
        # dK_ddists = (d_constant_component * exp_component) + (constant_component * d_exp_component)
        # d_exp_component = consts * exp_component
        V_dK_ddists = torch.add(d_constant_component, constant_component, alpha=-math.sqrt(self.nu * 2)).mul_(
            exp_component
        )
        if V is not None:
            V_dK_ddists.mul_(V)
        # dK_dXi = dK_ddists * ddists_dXi
        # ddists_dX1 = diffs / dists
        # ddists_dX1 = - diffs / dists
        V_dK_ddists_div_dists = V_dK_ddists.div_(dists).nan_to_num_(nan=0.0)
        X1_grad = torch.einsum("...MN,...MND->...MD", V_dK_ddists_div_dists, diffs)
        X2_grad = torch.einsum("...MN,...MND->...ND", V_dK_ddists_div_dists, diffs).mul_(-1)

        return K, (X1_grad, X2_grad)
