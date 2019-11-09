#!/usr/bin/env python3

import math
import torch
from .. import settings
from ..lazy import (
    DiagLazyTensor, CachedCGLazyTensor, CholLazyTensor, PsdSumLazyTensor,
    RootLazyTensor, ZeroLazyTensor, delazify
)
from ..distributions import MultivariateNormal
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.cholesky import psd_safe_cholesky
from ..utils.memoize import cached
from ._variational_strategy import _VariationalStrategy


class UnwhitenedVariationalStrategy(_VariationalStrategy):
    """
    VariationalStrategy objects control how certain aspects of variational inference should be performed. In particular,
    they define two methods that get used during variational inference:

    #. The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
       GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
       this is done simply by calling the user defined GP prior on the inducing point data directly.
    # The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
      inducing point function values. Specifically, forward defines how to transform a variational distribution
      over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
      specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

    In GPyTorch, we currently support two example instances of this latter functionality. In scenarios where the
    inducing points are learned or at least not constrained to a grid, we apply the derivation in Hensman et al., 2015
    to exactly marginalize out the variational distribution. When the inducing points are constrained to a grid, we
    apply the derivation in Wilson et al., 2016 and exploit a deterministic relationship between f and u.
    """
    @cached(name="cholesky_factor")
    def _cholesky_factor(self, induc_induc_covar):
        # Maybe used - if we're not using CG
        L = psd_safe_cholesky(delazify(induc_induc_covar))
        return L

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        out = self.model.forward(self.inducing_points)
        res = MultivariateNormal(
            out.mean, out.lazy_covariance_matrix.add_jitter()
        )
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

        # If we're less than a certain size, we'll compute the Cholesky decomposition of induc_induc_covar
        cholesky = False
        if settings.fast_computations.log_prob.off() or (num_induc <= settings.max_cholesky_size.value()):
            induc_induc_covar = CholLazyTensor(self._cholesky_factor(induc_induc_covar))
            cholesky = True

        # If we are making predictions and don't need variances, we can do things very quickly.
        if not self.training and settings.skip_posterior_variances.on():
            if not hasattr(self, "_mean_cache"):
                # For now: run variational inference without a preconditioner
                # The preconditioner screws things up for some reason
                with settings.max_preconditioner_size(0):
                    self._mean_cache = induc_induc_covar.inv_matmul(mean_diff).detach()
            predictive_mean = torch.add(
                test_mean,
                induc_data_covar.transpose(-2, -1).matmul(self._mean_cache).squeeze(-1)
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

        # Cache the CG results
        # For now: run variational inference without a preconditioner
        # The preconditioner screws things up for some reason
        with settings.max_preconditioner_size(0):
            # Cache the CG results
            if variational_inducing_covar is None:
                left_tensors = mean_diff
            else:
                left_tensors = torch.cat([mean_diff, root_variational_covar], -1)

            with torch.no_grad():
                eager_rhs = torch.cat([left_tensors, induc_data_covar], -1)
                solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLazyTensor.precompute_terms(
                    induc_induc_covar, eager_rhs.detach(), logdet_terms=(not cholesky),
                    include_tmats=(not settings.skip_logdet_forward.on() and not cholesky)
                )
                eager_rhss = [
                    eager_rhs.detach(), eager_rhs[..., left_tensors.size(-1):].detach(),
                    eager_rhs[..., :left_tensors.size(-1)].detach()
                ]
                solves = [
                    solve.detach(), solve[..., left_tensors.size(-1):].detach(),
                    solve[..., :left_tensors.size(-1)].detach()
                ]
                if settings.skip_logdet_forward.on():
                    eager_rhss.append(torch.cat([probe_vecs, left_tensors], -1))
                    solves.append(torch.cat([probe_vec_solves, solve[..., :left_tensors.size(-1)]], -1))
            induc_induc_covar = CachedCGLazyTensor(
                induc_induc_covar, eager_rhss=eager_rhss, solves=solves, probe_vectors=probe_vecs,
                probe_vector_norms=probe_vec_norms, probe_vector_solves=probe_vec_solves,
                probe_vector_tmats=tmats,
            )

        # Cache the kernel matrix with the cached CG calls
        if self.training:
            self._memoize_cache["prior_distribution_memo"] = MultivariateNormal(induc_mean, induc_induc_covar)

        # Compute predictive mean
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
                induc_data_covar.transpose(-1, -2).mul(-1),
                induc_induc_covar.inv_matmul(induc_data_covar)
            )
            data_covariance = data_data_covar + neg_induc_data_data_covar
        predictive_covar = PsdSumLazyTensor(RootLazyTensor(inv_products[..., 1:, :].transpose(-1, -2)), data_covariance)

        # Done!
        return MultivariateNormal(predictive_mean, predictive_covar)
