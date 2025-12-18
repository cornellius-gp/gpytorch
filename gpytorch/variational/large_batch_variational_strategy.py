#!/usr/bin/env python3

from typing import Optional

import torch
from linear_operator import to_dense
from linear_operator.operators import LinearOperator, MatmulLinearOperator, SumLinearOperator
from torch import Tensor

from gpytorch.variational.variational_strategy import VariationalStrategy

from ..distributions import MultivariateNormal
from ..settings import _linalg_dtype_cholesky
from ..utils.errors import CachingError
from ..utils.memoize import pop_from_cache_ignore_args


class LargeBatchVariationalStrategy(VariationalStrategy):
    r"""A lightweight and performant variational strategy that is optimized for large batch training.

    This class groups the middle term `K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2}` in double precision.
    """

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: Optional[LinearOperator] = None,
        **kwargs,
    ) -> MultivariateNormal:
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)
        induc_data_covar = full_covar[..., :num_induc, num_induc:].to_dense()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)

        # Need to make `L` dense because linear operators do not support triangular solves with `left=False`
        L = to_dense(L)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        vec = torch.linalg.solve_triangular(
            L.mT.type(full_inputs.dtype), inducing_values.unsqueeze(-1), upper=True, left=True
        )
        predictive_mean = (induc_data_covar.mT @ vec).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = to_dense(variational_inducing_covar) + to_dense(middle_term)

        middle_term = middle_term.type(_linalg_dtype_cholesky.value())
        middle_term = torch.linalg.solve_triangular(L, middle_term, upper=False, left=False)
        middle_term = torch.linalg.solve_triangular(L.mT, middle_term, upper=True, left=True)

        right_term = middle_term @ induc_data_covar.type(_linalg_dtype_cholesky.value())
        right_term = right_term.type(full_inputs.dtype)

        predictive_covar = SumLinearOperator(
            data_data_covar.add_jitter(self.jitter_val),
            MatmulLinearOperator(induc_data_covar.transpose(-1, -2), right_term),
        )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
