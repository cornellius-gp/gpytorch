#!/usr/bin/env python3

import torch
from .lazy_tensor import LazyTensor, delazify
from .non_lazy_tensor import lazify
from .. import settings


class SymmetricKernelInterpolatedLazyTensor(LazyTensor):
    def __init__(self, base_lazy_tensor, induc_induc_covar, data_induc_covar):
        base_lazy_tensor = lazify(base_lazy_tensor)
        induc_induc_covar = lazify(induc_induc_covar)
        data_induc_covar = delazify(data_induc_covar)

        if settings.debug.on():
            if induc_induc_covar.shape != base_lazy_tensor.shape:
                raise ValueError(
                    "induc_induc_covar {} and base_lazy_tensor {} should have same "
                    "shapes.".format(induc_induc_covar.shape, base_lazy_tensor.shape)
                )
            if (induc_induc_covar.shape[:-2] != data_induc_covar.shape[:-2]) or \
                    (induc_induc_covar.size(-1) != data_induc_covar.size(-1)):
                raise ValueError(
                    "induc_induc_covar {} and data_induc_covar {} should have compatible "
                    "shapes.".format(induc_induc_covar.shape, base_lazy_tensor.shape)
                )

        super(SymmetricKernelInterpolatedLazyTensor, self).__init__(
            base_lazy_tensor, induc_induc_covar, data_induc_covar,
        )
        self.base_lazy_tensor = base_lazy_tensor
        self.induc_induc_covar = induc_induc_covar
        self.data_induc_covar = data_induc_covar

    @property
    def _kernel_interpolation_term(self):
        if not hasattr(self, "_kernel_interpolation_term_memo"):
            self._kernel_interpolation_term_memo = self.induc_induc_covar.inv_matmul(self.induc_data_covar)
        return self._kernel_interpolation_term_memo

    @property
    def induc_data_covar(self):
        if not hasattr(self, "_induc_data_covar_memo"):
            self._induc_data_covar_memo = self.data_induc_covar.transpose(-1, -2)
        return self._induc_data_covar_memo

    def _get_indices(self, row_indices, col_indices, *batch_indices):
        left_interp_tensor = self._kernel_interpolation_term[(*batch_indices, slice(None, None, None), row_indices)]
        right_interp_tensor = self._kernel_interpolation_term[(*batch_indices, slice(None, None, None), col_indices)]
        base_lazy_tensor = self.base_lazy_tensor[(*batch_indices, slice(None, None, None), slice(None, None, None))]

        squeeze = False
        if right_interp_tensor.dim() < base_lazy_tensor.dim():
            right_interp_tensor = right_interp_tensor.unsqueeze(-1)
            squeeze = True
        if left_interp_tensor.dim() < base_lazy_tensor.dim():
            left_interp_tensor = left_interp_tensor.unsqueeze(-1)
            squeeze = True

        res = (base_lazy_tensor @ right_interp_tensor * left_interp_tensor).sum(-2)
        if squeeze:
            res = res.squeeze(-1)
        return res

    def _quad_form_derivative(self, left_factor, right_factor):
        # K^-1 K_x left_factor, K^-1 K_x right_factor
        interp_left_factor = self.induc_induc_covar.inv_matmul(self.induc_data_covar @ left_factor)
        interp_right_factor = self.induc_induc_covar.inv_matmul(self.induc_data_covar @ right_factor)

        # K^-1 S K^-1 K_x left_factor, K^-1 S K^-1 K_x right_factor
        double_interp_term = self.induc_induc_covar.inv_matmul(
            self.base_lazy_tensor @ self._kernel_interpolation_term
        )
        double_interp_left_factor = double_interp_term @ left_factor
        double_interp_right_factor = double_interp_term @ right_factor

        # Actual derivatives
        base_lazy_tensor_deriv = self.base_lazy_tensor._quad_form_derivative(interp_left_factor, interp_right_factor)
        induc_induc_covar_deriv = self.induc_induc_covar._quad_form_derivative(
            torch.cat([double_interp_right_factor, interp_right_factor], -1),
            torch.cat([interp_left_factor, double_interp_left_factor], -1).mul_(-1),
        )
        data_induc_covar_deriv = torch.add(
            left_factor @ double_interp_right_factor.transpose(-1, -2),
            right_factor @ double_interp_left_factor.transpose(-1, -2),
        )

        return base_lazy_tensor_deriv + induc_induc_covar_deriv + (data_induc_covar_deriv,)

    def _matmul(self, rhs):
        res = self._kernel_interpolation_term @ rhs
        res = self.base_lazy_tensor.matmul(res)
        return self._kernel_interpolation_term.transpose(-1, -2).matmul(
            self.base_lazy_tensor @ (self._kernel_interpolation_term @ rhs)
        )

    def _size(self):
        *batch_shape, _, data_size = self.induc_data_covar.shape
        return torch.Size((*batch_shape, data_size, data_size))

    def _transpose_nonbatch(self):
        # Matrix is symmetric
        return self

    def zero_mean_mvn_samples(self, num_samples, samples_dim=0):
        base_samples = self.base_lazy_tensor.zero_mean_mvn_samples(num_samples, samples_dim=-1)
        samples = self.induc_induc_covar.inv_matmul(base_samples, left_tensor=self.data_induc_covar)

        if samples_dim == 0:
            return samples.permute(-1, *range(self.dim() - 1)).contiguous()
        elif samples_dim == -1:
            return samples
        else:
            raise RuntimeError(
                "LazyTensor.zero_mean_mvn_samples expects samples_dim=0 or samples_dim=-1. Got {}".format(samples_dim)
            )
