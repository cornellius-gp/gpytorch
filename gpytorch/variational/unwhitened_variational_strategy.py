#!/usr/bin/env python3

import math

import torch

from .. import settings
from ..distributions import MultivariateNormal
from ..lazy import (
    CholLazyTensor,
    DiagLazyTensor,
    PsdSumLazyTensor,
    RootLazyTensor,
    TriangularLazyTensor,
    ZeroLazyTensor,
    delazify,
)
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.cholesky import psd_safe_cholesky
from ..utils.memoize import add_to_cache, cached
from ._variational_strategy import _VariationalStrategy


class UnwhitenedVariationalStrategy(_VariationalStrategy):
    r"""
    Similar to :obj:`~gpytorch.variational.VariationalStrategy`, but does not perform the
    whitening operation. In almost all cases :obj:`~gpytorch.variational.VariationalStrategy`
    is preferable, with a few exceptions:

    - When the inducing points are exactly equal to the training points (i.e. :math:`\mathbf Z = \mathbf X`).
      Unwhitened models are faster in this case.

    - When the number of inducing points is very large (e.g. >2000). Unwhitened models can use CG for faster
      computation.

    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param torch.Tensor inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param bool learn_inducing_points: (optional, default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    """

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar):
        # Maybe used - if we're not using CG
        L = psd_safe_cholesky(delazify(induc_induc_covar))
        return TriangularLazyTensor(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        out = self.model.forward(self.inducing_points)
        res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter())
        return res

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None):
        # If our points equal the inducing points, we're done
        if torch.equal(x, inducing_points):
            if variational_inducing_covar is None:
                raise RuntimeError
            else:
                return MultivariateNormal(inducing_values, variational_inducing_covar)

        # Otherwise, we have to marginalize
        num_induc = inducing_points.size(-2)
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

        # Mean terms
        test_mean = full_mean[..., num_induc:]
        induc_mean = full_mean[..., :num_induc]
        mean_diff = (inducing_values - induc_mean).unsqueeze(-1)

        # Covariance terms
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute Cholesky factorization of inducing covariance matrix
        if settings.fast_computations.log_prob.off() or (num_induc <= settings.max_cholesky_size.value()):
            induc_induc_covar = CholLazyTensor(self._cholesky_factor(induc_induc_covar))

        # If we are making predictions and don't need variances, we can do things very quickly.
        if not self.training and settings.skip_posterior_variances.on():
            self._mean_cache = induc_induc_covar.inv_matmul(mean_diff).detach()
            predictive_mean = torch.add(
                test_mean, induc_data_covar.transpose(-2, -1).matmul(self._mean_cache).squeeze(-1)
            )
            predictive_covar = ZeroLazyTensor(test_mean.size(-1), test_mean.size(-1))
            return MultivariateNormal(predictive_mean, predictive_covar)

        # Expand everything to the right size
        shapes = [mean_diff.shape[:-1], induc_data_covar.shape[:-1], induc_induc_covar.shape[:-1]]
        if variational_inducing_covar is not None:
            root_variational_covar = variational_inducing_covar.root_decomposition().root.evaluate()
            shapes.append(root_variational_covar.shape[:-1])
        shape = _mul_broadcast_shape(*shapes)
        mean_diff = mean_diff.expand(*shape, mean_diff.size(-1))
        induc_data_covar = induc_data_covar.expand(*shape, induc_data_covar.size(-1))
        induc_induc_covar = induc_induc_covar.expand(*shape, induc_induc_covar.size(-1))
        if variational_inducing_covar is not None:
            root_variational_covar = root_variational_covar.expand(*shape, root_variational_covar.size(-1))

        # Cache the kernel matrix with the cached CG calls
        if self.training:
            prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)
            add_to_cache(self, "prior_distribution_memo", prior_dist)

        # Compute predictive mean
        if variational_inducing_covar is None:
            left_tensors = mean_diff
        else:
            left_tensors = torch.cat([mean_diff, root_variational_covar], -1)
        inv_products = induc_induc_covar.inv_matmul(induc_data_covar, left_tensors.transpose(-1, -2))
        predictive_mean = torch.add(test_mean, inv_products[..., 0, :])

        # Compute covariance
        if self.training:
            interp_data_data_var, _ = induc_induc_covar.inv_quad_logdet(
                induc_data_covar, logdet=False, reduce_inv_quad=False
            )
            data_covariance = DiagLazyTensor((data_data_covar.diag() - interp_data_data_var).clamp(0, math.inf))
        else:
            neg_induc_data_data_covar = torch.matmul(
                induc_data_covar.transpose(-1, -2).mul(-1), induc_induc_covar.inv_matmul(induc_data_covar)
            )
            data_covariance = data_data_covar + neg_induc_data_data_covar
        predictive_covar = PsdSumLazyTensor(RootLazyTensor(inv_products[..., 1:, :].transpose(-1, -2)), data_covariance)

        # Done!
        return MultivariateNormal(predictive_mean, predictive_covar)
