"""Computation-aware Gaussian process."""

import math
from typing import Union

import torch
from jaxtyping import Float

from linear_operator import operators, utils as linop_utils
from torch import Tensor

from .. import kernels, likelihoods, means, settings

from ..distributions import MultivariateNormal
from .exact_gp import ExactGP


class _BlockDiagonalSparseLinearOperator(operators.LinearOperator):
    """A sparse linear operator (which when reordered) has dense blocks on its diagonal.

    Linear operator with a matrix representation that has sparse rows, with an equal number of
    non-zero entries per row. The non-zero entries are stored in a tensor of size M x NNZ, where M is
    the number of rows and NNZ is the number of non-zero entries per row. When appropriately re-ordering
    the columns of the matrix, it is a block-diagonal matrix.

    Note:
        This currently only supports equally sized blocks of size 1 x NNZ.

    :param non_zero_idcs: Tensor of non-zero indices.
    :param blocks: Tensor of non-zero entries.
    :param size_input_dim: Size of the (sparse) input dimension, equivalently the number of columns.
    """

    # NOTE: Since linear_operator currently primarily supports square linear operators and the
    # functionality of this class is only used in "ComputationAwareGP", this class is defined locally.

    def __init__(
        self,
        non_zero_idcs: Float[torch.Tensor, "M NNZ"],  # noqa F722
        blocks: Float[torch.Tensor, "M NNZ"],  # noqa F722
        size_input_dim: int,
    ):
        super().__init__(non_zero_idcs, blocks, size_input_dim=size_input_dim)
        self.non_zero_idcs = torch.atleast_2d(non_zero_idcs)
        self.non_zero_idcs.requires_grad = False  # Ensure indices cannot be optimized
        self.blocks = torch.atleast_2d(blocks)
        self.size_input_dim = size_input_dim

    def _matmul(
        self: Float[operators.LinearOperator, "*batch M N"],  # noqa F722
        rhs: Float[torch.Tensor, "*batch2 N C"],  # noqa F722
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:  # noqa F722
        # Workarounds for (Added)DiagLinearOperator
        if isinstance(rhs, operators.AddedDiagLinearOperator):
            return self._matmul(rhs._linear_op) + self._matmul(rhs._diag_tensor)

        # There seems to be a bug in DiagLinearOperator, which doesn't allow subsetting the way we do here.
        if isinstance(rhs, operators.DiagLinearOperator):
            return _BlockDiagonalSparseLinearOperator(
                non_zero_idcs=self.non_zero_idcs,
                blocks=rhs.diag()[self.non_zero_idcs] * self.blocks,
                size_input_dim=self.size_input_dim,
            ).to_dense()

        # Subset rhs via index tensor
        rhs_non_zero = rhs[..., self.non_zero_idcs, :]

        # Multiply on sparse dimension
        return (self.blocks.unsqueeze(-2) @ rhs_non_zero).squeeze(-2)

    def _size(self) -> torch.Size:
        return torch.Size((self.non_zero_idcs.shape[0], self.size_input_dim))

    def to_dense(self: operators.LinearOperator) -> Tensor:
        if self.size() == self.blocks.shape:
            return self.blocks
        return torch.zeros(
            (self.blocks.shape[0], self.size_input_dim), dtype=self.blocks.dtype, device=self.blocks.device
        ).scatter_(src=self.blocks, index=self.non_zero_idcs, dim=1)


class ComputationAwareGP(ExactGP):
    """Computation-aware Gaussian process.

    A scalable Gaussian process, which captures the inevitable approximation error as additional
    uncertainty -- enabling an explicit tradeoff between computational efficiency and precision.

    The method implemented here projects the data onto a learned sparse basis as proposed by
    `Wenger et al. (2024)`_, leading to linear-time inference and model selection.

    .. _Wenger et al. (2022):
        https://arxiv.org/abs/2205.15449
    .. _Wenger et al. (2024):
        https://arxiv.org/abs/2411.01036

    :param train_inputs: Training inputs.
    :param train_targets: Training targets.
    :param likelihood: Likelihood.
    :param projection_dim: Dimension of the lower-dimensional space which the data is projected onto.
    :param initialization: Initialization of the action entries. Default is 'random'.

    Example:
        >>> from gpytorch import models, means, kernels, likelihoods, distributions
        ...
        >>> class CaGP(models.ComputationAwareGP):
        >>>     def __init__(self, train_inputs, train_targets, likelihood, projection_dim):
        >>>         super().__init__(train_inputs, train_targets, likelihood, projection_dim=projection_dim)
        >>>         self.mean_module = means.ZeroMean()
        >>>         self.covar_module = kernels.ScaleKernel(kernels.MaternKernel(nu=1.5))
        >>>
        >>>     def forward(self, x):
        >>>         mean = self.mean_module(x)
        >>>         covar = self.covar_module(x)
        >>>         return distributions.MultivariateNormal(mean, covar)
        >>>
        >>> # train_inputs = ...; train_targets = ...
        >>> likelihood = likelihoods.GaussianLikelihood()
        >>> model = CaGP(train_inputs, train_targets, likelihood, projection_dim=...)
        >>>
        >>> # test_x = ...;
        >>> model(test_x)  # Returns the GP latent function at test_x
        >>> likelihood(model(test_x))  # Returns the (approximate) predictive posterior distribution at test_x
    """

    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        mean_module: "means.Mean",
        covar_module: "kernels.Kernel",
        likelihood: "likelihoods.GaussianLikelihood",
        projection_dim: int,
        initialization: str = "random",
    ):

        # Set number of non-zero action entries such that num_non_zero * projection_dim = num_train_targets
        num_non_zero = train_targets.size(-1) // projection_dim

        super().__init__(
            # Training data is subset to satisfy the requirement: num_non_zero * projection_dim = num_train_targets
            train_inputs[0 : num_non_zero * projection_dim],
            train_targets[0 : num_non_zero * projection_dim],
            likelihood,
        )
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.projection_dim = projection_dim
        self.num_non_zero = num_non_zero
        self.cholfac_gram_SKhatS = None

        non_zero_idcs = torch.arange(
            self.num_non_zero * projection_dim,
            device=train_inputs.device,
        ).reshape(self.projection_dim, -1)

        if initialization == "random":
            # Random initialization of actions
            self.non_zero_action_entries = torch.nn.Parameter(
                torch.randn_like(
                    non_zero_idcs,
                    dtype=train_inputs.dtype,
                    device=train_inputs.device,
                ).div(math.sqrt(self.num_non_zero))
            )
        else:
            raise ValueError(f"Unknown initialization: '{initialization}'.")

        self.actions_op = _BlockDiagonalSparseLinearOperator(
            non_zero_idcs=non_zero_idcs,
            blocks=self.non_zero_action_entries,
            size_input_dim=self.projection_dim * self.num_non_zero,
        )

    def __call__(self, x: torch.Tensor) -> MultivariateNormal:
        # TODO: remove usage of mean_module and covar_module and replace with "forward"
        # TODO: remove explicit calling of kernel_forward_fn and lengthscale

        if self.training:
            # In training mode, just return the prior.
            return MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )
        elif settings.prior_mode.on():
            # Prior mode
            return MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )
        else:
            # Posterior mode
            if x.ndim == 1:
                x = torch.atleast_2d(x).mT

            # Kernel forward and hyperparameters
            if isinstance(self.covar_module, kernels.ScaleKernel):
                outputscale = self.covar_module.outputscale
                lengthscale = self.covar_module.base_kernel.lengthscale
                kernel_forward_fn = self.covar_module.base_kernel._forward_no_kernel_linop
            else:
                outputscale = 1.0
                lengthscale = self.covar_module.lengthscale
                kernel_forward_fn = self.covar_module._forward_no_kernel_linop

            if self.cholfac_gram_SKhatS is None:
                # If the Cholesky factor of the gram matrix S'(K + noise)S hasn't been precomputed
                # (in the loss function), compute it.
                K_lazy = kernel_forward_fn(
                    self.train_inputs[0]
                    .div(lengthscale)
                    .view(self.projection_dim, self.num_non_zero, self.train_inputs[0].shape[-1]),
                    self.train_inputs[0]
                    .div(lengthscale)
                    .view(self.projection_dim, 1, self.num_non_zero, self.train_inputs[0].shape[-1]),
                )
                gram_SKS = (
                    (
                        (K_lazy @ self.actions_op.blocks.view(self.projection_dim, 1, self.num_non_zero, 1)).squeeze(-1)
                        * self.actions_op.blocks
                    )
                    .sum(-1)
                    .mul(outputscale)
                )

                StrS_diag = (self.actions_op.blocks**2).sum(-1)  # NOTE: Assumes orthogonal actions.
                gram_SKhatS = gram_SKS + torch.diag(self.likelihood.noise * StrS_diag)
                self.cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
                    gram_SKhatS.to(dtype=torch.float64), upper=False
                )

            # Cross-covariance mapped to the low-dimensional space spanned by the actions: k(x, X)S
            covar_x_train_actions = (
                (
                    kernel_forward_fn(
                        x / lengthscale,
                        (self.train_inputs[0] / lengthscale).view(
                            self.projection_dim, self.num_non_zero, self.train_inputs[0].shape[-1]
                        ),
                    )
                    @ self.actions_op.blocks.view(self.projection_dim, self.num_non_zero, 1)
                )
                .squeeze(-1)
                .mT.mul(outputscale)
            )

            # Matrix-square root of the covariance downdate: k(x, X)L^{-1}
            covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
                self.cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
            ).mT

            # "Projected" training data (with mean correction)
            actions_target = self.actions_op @ (self.train_targets - self.mean_module(self.train_inputs[0]))

            # Compressed representer weights
            compressed_repr_weights = (
                torch.cholesky_solve(
                    actions_target.unsqueeze(1).to(dtype=torch.float64), self.cholfac_gram_SKhatS, upper=False
                )
                .squeeze(-1)
                .to(self.train_inputs[0].dtype)
            )

            # (Combined) posterior mean and covariance evaluated at the test point(s)
            mean = self.mean_module(x) + covar_x_train_actions @ compressed_repr_weights
            covar = self.covar_module(x) - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)

            return MultivariateNormal(mean, covar)
