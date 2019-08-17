#!/usr/bin/env python3

#TODO: clean this up
import torch
import math
from .. import settings
from . import GridKernel

from ..lazy import InterpolatedLazyTensor, KroneckerProductLazyLogDet, NonLazyTensor, lazify, delazify, ToeplitzLazyTensor
from ..models.exact_prediction_strategies import InterpolatedPredictionStrategy
from ..utils.interpolation import Interpolation

class GridKronWrapper(GridKernel):

    def __init__(self, base_kernel, grid, interpolation_mode=False, active_dims=None):
        super(GridKronWrapper, self).__init__(base_kernel, grid, interpolation_mode, active_dims)
        self.base_kernel = base_kernel
        self.interpolation_mode = interpolation_mode
        self.log_det = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # print("forward being hit")
        grid = self.grid

        if not self.interpolation_mode:
            if len(x1.shape[:-2]):
                full_grid = self.full_grid.expand(*x1.shape[:-2], *self.full_grid.shape[-2:])
            else:
                full_grid = self.full_grid

        # print(x1)
        # print(x2)
        # print("toeplitz", settings.use_toeplitz.on())
        if self.interpolation_mode or (torch.equal(x1, full_grid) and torch.equal(x2, full_grid)):
            if not self.training and hasattr(self, "_cached_kernel_mat"):
                return self._cached_kernel_mat

            n_dim = x1.size(-1)

            if settings.use_toeplitz.on():
                covars = []
                for dim in range(n_dim):
                    temp = self.base_kernel.base_kernels[dim](grid[0, dim].unsqueeze(-1),
                                                              grid[:, dim]).evaluate().squeeze(0)
                    covars.append(ToeplitzLazyTensor(temp))


            #     first_item = grid[0:1]
            #     print("LOOK HERE")
            #     covar_columns = self.base_kernel(first_item, grid, diag=False, last_dim_is_batch=True, **params)
            #     print("covar_columns shape = ", covar_columns.shape)
            #     print("last_dim_is_batch", last_dim_is_batch)
            #     covar_columns = delazify(covar_columns).squeeze(-2)
            #
            #     if last_dim_is_batch:
            #         covars = [ToeplitzLazyTensor(covar_columns.squeeze(-2))]
            #     else:
            #         covars = [ToeplitzLazyTensor(covar_columns[i : i + 1].squeeze(-2)) for i in range(n_dim)]
            #         print("covars = ", covars)
            else:
                full_covar = self.base_kernel(grid, grid, last_dim_is_batch=True, **params)
                if last_dim_is_batch:
                    covars = [full_covar]
                else:
                    covars = [full_covar[i : i + 1].squeeze(0) for i in range(n_dim)]
            if len(covars) > 1:
                # print("calling kronecker")
                # print([cv.evaluate().shape for cv in covars])
                covar = KronLazyLogDet(*covars[::-1])
            else:
                covar = covars[0]

            if not self.training:
                self._cached_kernel_mat = covar
            # print("fwd returning ", covar)
            return covar
        else:
            return self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch)

    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        # print("custom __call__")
        x1_, x2_ = x1, x2

        # Select the active dimensions
        if self.active_dims is not None:
            x1_ = x1_.index_select(-1, self.active_dims)
            if x2_ is not None:
                x2_ = x2_.index_select(-1, self.active_dims)

        # Give x1_ and x2_ a last dimension, if necessary
        if x1_.ndimension() == 1:
            x1_ = x1_.unsqueeze(1)
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")

        if x2_ is None:
            x2_ = x1_

        return self.forward(x1_, x2_, diag=diag, last_dim_is_batch=last_dim_is_batch,
                        **params)

    def interp_predictions(self, test_x, dens_list=None):
        ## setting of latent densities ##
        if dens_list is not None:
            for idx, dens in enumerate(dens_list):
                if dens is not None:
                    self.set_latent_params(dens, idx)

        k_uu = self.base_kernel(self.grid, self.grid, last_dim_is_batch=True)

        left_interp_indices, left_interp_values = self._compute_grid(test_x)

        ## this might be the identity, not sure.
        right_interp_indices, right_interp_values = self._compute_grid(self.grid)

        



    def _compute_grid(self, inputs, last_dim_is_batch=False):
        n_data, n_dimensions = inputs.size(-2), inputs.size(-1)
        if last_dim_is_batch:
            inputs = inputs.transpose(-1, -2).unsqueeze(-1)
            n_dimensions = 1
        batch_shape = inputs.shape[:-2]

        inputs = inputs.contiguous().view(-1, n_dimensions)
        interp_indices, interp_values = Interpolation().interpolate(self.grid, inputs)
        interp_indices = interp_indices.view(*batch_shape, n_data, -1)
        interp_values = interp_values.view(*batch_shape, n_data, -1)
        return interp_indices, interp_values


    def get_latent_mod(self, idx=None):
        return self.base_kernel.base_kernels[idx].latent_mod

    def get_latent_lh(self, idx=None):
        return self.base_kernel.base_kernels[idx].latent_lh

    def get_omega(self, idx=None):
        return self.base_kernel.base_kernels[idx].omega

    def get_latent_params(self, idx=None):
        return self.base_kernel.base_kernels[idx].latent_params

    def set_latent_params(self, x, idx=None):
        self.base_kernel.base_kernels[idx].set_latent_params(x)
