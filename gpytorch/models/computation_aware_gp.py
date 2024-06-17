#!/usr/bin/env python3
"""Computation-aware Gaussian processes."""

import math

import torch
from linear_operator import operators, utils as linop_utils

from .. import kernels

from ..distributions import MultivariateNormal
from .exact_gp import ExactGP


class ComputationAwareGP(ExactGP):
    """Computation-aware Gaussian process."""

    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        mean,
        kernel,
        likelihood,
        num_iter: int,
        # num_non_zero: int,
        # chunk_size: Optional[int] = None,
        initialization: str = "random",
    ):

        num_non_zero = train_targets.size(-1) // num_iter
        super().__init__(
            train_inputs[0 : num_non_zero * num_iter], train_targets[0 : num_non_zero * num_iter], likelihood
        )
        self.mean_module = mean
        self.covar_module = kernel
        self.num_iter = num_iter
        self.num_non_zero = num_non_zero
        # self.chunk_size = chunk_size

        # # Random initialization
        # non_zero_idcs = torch.cat(
        #     [
        #         torch.randperm(num_train, device=train_inputs.device)[: self.num_non_zero].unsqueeze(-2)
        #         for _ in range(self.num_iter)
        #     ],
        #     dim=0,
        # )
        # blocks = torch.nn.Parameter(
        #     torch.randn(
        #         self.num_iter,
        #         self.num_non_zero,
        #         dtype=train_inputs.dtype,
        #         device=train_inputs.device,
        #     )
        # )
        # Initialize with rhs of linear system (i.e. train targets)
        non_zero_idcs = torch.arange(
            self.num_non_zero * num_iter,
            # int(math.sqrt(num_train / num_iter)) * num_iter,  # TODO: This does not include all training datapoints!
            device=train_inputs.device,
        ).reshape(self.num_iter, -1)
        # self.num_non_zero = non_zero_idcs.shape[1]

        if initialization == "random":
            self.non_zero_action_entries = torch.nn.Parameter(
                torch.randn_like(
                    non_zero_idcs,
                    dtype=train_inputs.dtype,
                    device=train_inputs.device,
                ).div(math.sqrt(self.num_non_zero))
            )
        elif initialization == "targets":
            self.non_zero_action_entries = torch.nn.Parameter(
                train_targets.clone()[: self.num_non_zero * num_iter].reshape(self.num_iter, -1)
                # train_targets.clone()[: int(math.sqrt(num_train / num_iter)) * num_iter].reshape(self.num_iter, -1)
            )
            self.non_zero_action_entries.div(
                torch.linalg.vector_norm(self.non_zero_action_entries, dim=1).reshape(-1, 1)
            )
        elif initialization == "eigen":
            with torch.no_grad():
                X = train_inputs.clone()[0 : num_non_zero * num_iter].reshape(
                    num_iter, num_non_zero, train_inputs.shape[-1]
                )
                K_sub_matrices = self.covar_module(X)
                _, evecs = torch.linalg.eigh(K_sub_matrices.to_dense())
            self.non_zero_action_entries = torch.nn.Parameter(evecs[:, -1])

        elif initialization == "fourier":
            # X = (
            #     train_inputs.clone()
            #     .detach()[0 : num_non_zero * num_iter]
            #     .reshape(num_iter, num_non_zero)
            #     .requires_grad_(False)
            # )
            X = torch.ones_like(train_inputs)[0 : num_non_zero * num_iter].reshape(num_iter, num_non_zero)
            self.frequencies = torch.nn.Parameter(
                torch.randn(
                    num_iter,
                    dtype=train_inputs.dtype,
                    device=train_inputs.device,
                )
            )
            self.offset = torch.nn.Parameter(
                2
                * math.pi
                * torch.rand(
                    num_iter,
                    dtype=train_inputs.dtype,
                    device=train_inputs.device,
                )
            )
            self.non_zero_action_entries = torch.cos(self.frequencies.reshape(-1, 1) * X + self.offset.reshape(-1, 1))
            # TODO: fails likely because BlockSparseLinearOperator also inherits from nn.Module
        else:
            raise ValueError(f"Unknown initialization: '{initialization}'.")

        self.actions = (
            operators.BlockSparseLinearOperator(  # TODO: Can we speed this up by allowing ranges as non-zero indices?
                non_zero_idcs=non_zero_idcs,
                blocks=self.non_zero_action_entries,
                size_sparse_dim=self.num_iter * self.num_non_zero,
            )
        )
        self.cholfac_gram_SKhatS = None

    def __call__(self, x):
        if self.training:
            return MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )
        else:
            # covar_train_train = self.covar_module(self.train_inputs[0])

            kernel = self.covar_module

            if isinstance(kernel, kernels.ScaleKernel):
                outputscale = kernel.outputscale
                lengthscale = kernel.base_kernel.lengthscale
                kernel_forward_fn = kernel.base_kernel._forward_no_kernel_linop
                base_kernel = kernel.base_kernel
            else:
                outputscale = 1.0
                lengthscale = kernel.lengthscale
                kernel_forward_fn = kernel._forward_no_kernel_linop
                base_kernel = kernel

            actions_op = self.actions

            # gram_SKS = kernels.SparseQuadForm.apply(
            #     self.train_inputs[0] / lengthscale,
            #     actions_op.blocks,
            #     # actions_op.non_zero_idcs.mT,
            #     None,
            #     kernel_forward_fn,
            #     None,
            #     self.chunk_size,
            # )
            if self.cholfac_gram_SKhatS is None:
                K_lazy = kernel_forward_fn(
                    self.train_inputs[0]
                    .div(lengthscale)
                    .view(self.num_iter, self.num_non_zero, self.train_inputs[0].shape[-1]),
                    self.train_inputs[0]
                    .div(lengthscale)
                    .view(self.num_iter, 1, self.num_non_zero, self.train_inputs[0].shape[-1]),
                )
                gram_SKS = (
                    (
                        (K_lazy @ actions_op.blocks.view(self.num_iter, 1, self.num_non_zero, 1)).squeeze(-1)
                        * actions_op.blocks
                    )
                    .sum(-1)
                    .mul(outputscale)
                )

                StrS_diag = (actions_op.blocks**2).sum(-1)  # NOTE: Assumes orthogonal actions.
                gram_SKhatS = gram_SKS + torch.diag(self.likelihood.noise * StrS_diag)
                self.cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
                    gram_SKhatS.to(dtype=torch.float64), upper=False
                )

            if x.ndim == 1:
                x = torch.atleast_2d(x).mT
            # covar_x_train_actions = outputscale * kernels.SparseLinearForm.apply(
            #     x / lengthscale,
            #     self.train_inputs[0] / lengthscale,
            #     actions_op.blocks,
            #     actions_op.non_zero_idcs,
            #     kernel_forward_fn,
            #     None,
            #     # self.chunk_size,  # TODO: the chunk size should probably be larger here, since we usually compute this on the test set
            # )
            covar_x_train_actions = (
                (
                    kernel_forward_fn(
                        x / lengthscale,
                        (self.train_inputs[0] / lengthscale).view(
                            self.num_iter, self.num_non_zero, self.train_inputs[0].shape[-1]
                        ),
                    )
                    @ actions_op.blocks.view(self.num_iter, self.num_non_zero, 1)
                )
                .squeeze(-1)
                .mT.mul(outputscale)
            )
            covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
                self.cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
            ).mT

            # Compressed representer weights
            actions_target = self.actions @ (self.train_targets - self.mean_module(self.train_inputs[0]))
            compressed_repr_weights = (
                torch.cholesky_solve(
                    actions_target.unsqueeze(1).to(dtype=torch.float64), self.cholfac_gram_SKhatS, upper=False
                )
                .squeeze(-1)
                .to(self.train_inputs[0].dtype)
            )

            mean = self.mean_module(x) + covar_x_train_actions @ compressed_repr_weights
            covar = self.covar_module(x) - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)

            return MultivariateNormal(mean, covar)

    # def __call__(self, x):
    #     if self.training:
    #         return MultivariateNormal(
    #             self.mean_module(x),
    #             self.covar_module(x),
    #         )
    #     else:
    #         # covar_train_train = self.covar_module(self.train_inputs[0])

    #         kernel = self.covar_module

    #         with torch.no_grad():
    #             outputscale = 1.0
    #             lengthscale = 1.0

    #             if isinstance(kernel, kernels.ScaleKernel):
    #                 outputscale = kernel.outputscale
    #                 lengthscale = kernel.base_kernel.lengthscale
    #                 forward_fn = kernel.base_kernel._forward
    #             else:
    #                 lengthscale = kernel.lengthscale

    #                 forward_fn = kernel._forward

    #         actions_op = self.actions

    #         gram_SKS, StrS = kernels.SparseBilinearForms.apply(
    #             self.train_inputs[0] / lengthscale,
    #             actions_op.blocks.mT,
    #             actions_op.blocks.mT,
    #             actions_op.non_zero_idcs.mT,
    #             actions_op.non_zero_idcs.mT,
    #             forward_fn,
    #             None,
    #             self.chunk_size,
    #         )  # TODO: Can compute StrS more efficiently since we assume actions are made up of non-intersecting blocks.
    #         gram_SKhatS = outputscale * gram_SKS + self.likelihood.noise * StrS

    #         if x.ndim == 1:
    #             x = torch.atleast_2d(x).mT
    #         covar_x_train_actions = outputscale * kernels.SparseLinearForms.apply(
    #             x / lengthscale,
    #             self.train_inputs[0] / lengthscale,
    #             actions_op.blocks,
    #             actions_op.non_zero_idcs,
    #             forward_fn,
    #             None,
    #             self.chunk_size,  # TODO: the chunk size should probably be larger here, since we usually compute this on the test set
    #         )

    #         cholfac_gram_SKhatS = linop_utils.cholesky.psd_safe_cholesky(
    #             gram_SKhatS.to(dtype=torch.float64), upper=False
    #         )
    #         covar_x_train_actions_cholfac_inv = torch.linalg.solve_triangular(
    #             cholfac_gram_SKhatS, covar_x_train_actions.mT, upper=False
    #         ).mT

    #         # Compressed representer weights
    #         actions_target = self.actions @ (self.train_targets - self.mean_module(self.train_inputs[0]))
    #         compressed_repr_weights = (
    #             torch.cholesky_solve(
    #                 actions_target.unsqueeze(1).to(dtype=torch.float64), cholfac_gram_SKhatS, upper=False
    #             )
    #             .squeeze(-1)
    #             .to(self.train_inputs[0].dtype)
    #         )

    #         mean = self.mean_module(x) + covar_x_train_actions @ compressed_repr_weights
    #         covar = self.covar_module(x) - operators.RootLinearOperator(root=covar_x_train_actions_cholfac_inv)

    #         return MultivariateNormal(mean, covar)
