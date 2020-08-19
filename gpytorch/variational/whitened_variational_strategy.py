#!/usr/bin/env python3

import math
import warnings

import torch

from .. import settings
from ..distributions import MultivariateNormal
from ..lazy import (
    BatchRepeatLazyTensor,
    CachedCGLazyTensor,
    CholLazyTensor,
    DiagLazyTensor,
    MatmulLazyTensor,
    PsdSumLazyTensor,
    RootLazyTensor,
    TriangularLazyTensor,
)
from ..module import Module
from ..utils.memoize import add_to_cache, cached, clear_cache_hook
from .unwhitened_variational_strategy import UnwhitenedVariationalStrategy


# Remove after 1.0
class WhitenedVariationalStrategy(UnwhitenedVariationalStrategy):
    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=True):
        warnings.warn(
            "WhitenedVariationalStrategy is deprecated. Please use VariationalStrategy instead.", DeprecationWarning
        )
        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations)

    @cached(name="logdet_memo")
    def prior_covar_logdet(self):
        return -self.prior_distribution.lazy_covariance_matrix.logdet()

    @cached(name="covar_trace_memo")
    def covar_trace(self):
        variational_covar = self.variational_distribution.covariance_matrix
        prior_covar = self.prior_distribution.covariance_matrix
        batch_shape = prior_covar.shape[:-2]
        return (variational_covar * prior_covar).view(*batch_shape, -1).sum(-1)

    @cached(name="mean_diff_inv_quad_memo")
    def mean_diff_inv_quad(self):
        prior_mean = self.prior_distribution.mean
        prior_covar = self.prior_distribution.lazy_covariance_matrix
        variational_mean = self.variational_distribution.mean
        return prior_covar.inv_quad(variational_mean - prior_mean)

    def kl_divergence(self):
        variational_dist_u = self.variational_distribution
        prior_dist = self.prior_distribution
        kl_divergence = 0.5 * sum(
            [
                # log|k| - log|S|
                # = log|K| - log|K var_dist_covar K|
                # = -log|K| - log|var_dist_covar|
                self.prior_covar_logdet(),
                -variational_dist_u.lazy_covariance_matrix.logdet(),
                # tr(K^-1 S) = tr(K^1 K var_dist_covar K) = tr(K var_dist_covar)
                self.covar_trace(),
                # (m - \mu u)^T K^-1 (m - \mu u)
                # = (K^-1 (m - \mu u)) K (K^1 (m - \mu u))
                # = (var_dist_mean)^T K (var_dist_mean)
                self.mean_diff_inv_quad(),
                # d
                -prior_dist.event_shape.numel(),
            ]
        )

        return kl_divergence

    def initialize_variational_dist(self):
        prior_dist = self.prior_distribution
        inv_prior_dist = torch.distributions.MultivariateNormal(
            prior_dist.mean,
            prior_dist.lazy_covariance_matrix.add_jitter()
            .evaluate()
            .double()
            .inverse()
            .type_as(prior_dist.covariance_matrix),
        )
        self.variational_distribution.initialize_variational_distribution(inv_prior_dist)

    def forward(self, x):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        :param torch.Tensor x: Locations x to get the variational posterior of the function values at.
        :rtype: ~gpytorch.distributions.MultivariateNormal
        :return: The distribution :math:`q(f|x)`
        """
        variational_dist = self.variational_distribution
        inducing_points = self.inducing_points
        if inducing_points.dim() < x.dim():
            inducing_points = inducing_points.expand(*x.shape[:-2], *inducing_points.shape[-2:])
        if len(variational_dist.batch_shape) < x.dim() - 2:
            variational_dist = variational_dist.expand(x.shape[:-2])

        # If our points equal the inducing points, we're done
        if torch.equal(x, inducing_points):
            # De-whiten the prior covar
            prior_covar = self.prior_distribution.lazy_covariance_matrix
            if isinstance(variational_dist.lazy_covariance_matrix, RootLazyTensor):
                predictive_covar = RootLazyTensor(prior_covar @ variational_dist.lazy_covariance_matrix.root.evaluate())
            else:
                predictive_covar = MatmulLazyTensor(prior_covar @ variational_dist.covariance_matrix, prior_covar)

            # Cache some values for the KL divergence
            if self.training:
                self._mean_diff_inv_quad_memo, self._logdet_memo = prior_covar.inv_quad_logdet(
                    (variational_dist.mean - self.prior_distribution.mean), logdet=True
                )

            return MultivariateNormal(variational_dist.mean, predictive_covar)

        # Otherwise, we have to marginalize
        else:
            num_induc = inducing_points.size(-2)
            full_inputs = torch.cat([inducing_points, x], dim=-2)
            full_output = self.model.forward(full_inputs)
            full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

            # Mean terms
            test_mean = full_mean[..., num_induc:]
            induc_mean = full_mean[..., :num_induc]
            mean_diff = (variational_dist.mean - induc_mean).unsqueeze(-1)

            # Covariance terms
            induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
            induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
            data_data_covar = full_covar[..., num_induc:, num_induc:]

            # If we're less than a certain size, we'll compute the Cholesky decomposition of induc_induc_covar
            cholesky = False
            if settings.fast_computations.log_prob.off() or (num_induc <= settings.max_cholesky_size.value()):
                induc_induc_covar = CholLazyTensor(TriangularLazyTensor(induc_induc_covar.cholesky()))
                cholesky = True

            # Cache the CG results
            # Do not use preconditioning for whitened VI, as it does not seem to improve performance.
            with settings.max_preconditioner_size(0):
                with torch.no_grad():
                    eager_rhs = torch.cat([induc_data_covar, mean_diff], -1)
                    solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLazyTensor.precompute_terms(
                        induc_induc_covar,
                        eager_rhs.detach(),
                        logdet_terms=(not cholesky),
                        include_tmats=(not settings.skip_logdet_forward.on() and not cholesky),
                    )
                    eager_rhss = [eager_rhs.detach()]
                    solves = [solve.detach()]
                    if settings.skip_logdet_forward.on() and self.training:
                        eager_rhss.append(torch.cat([probe_vecs, eager_rhs], -1))
                        solves.append(torch.cat([probe_vec_solves, solve[..., : eager_rhs.size(-1)]], -1))
                    elif not self.training:
                        eager_rhss.append(eager_rhs[..., :-1])
                        solves.append(solve[..., :-1])

                induc_induc_covar = CachedCGLazyTensor(
                    induc_induc_covar,
                    eager_rhss=eager_rhss,
                    solves=solves,
                    probe_vectors=probe_vecs,
                    probe_vector_norms=probe_vec_norms,
                    probe_vector_solves=probe_vec_solves,
                    probe_vector_tmats=tmats,
                )

            # Compute some terms that will be necessary for the predicitve covariance and KL divergence
            if self.training:
                interp_data_data_var_plus_mean_diff_inv_quad, logdet = induc_induc_covar.inv_quad_logdet(
                    torch.cat([induc_data_covar, mean_diff], -1), logdet=True, reduce_inv_quad=False
                )
                interp_data_data_var = interp_data_data_var_plus_mean_diff_inv_quad[..., :-1]
                mean_diff_inv_quad = interp_data_data_var_plus_mean_diff_inv_quad[..., -1]

            # Compute predictive mean
            predictive_mean = torch.add(
                test_mean,
                induc_induc_covar.inv_matmul(mean_diff, left_tensor=induc_data_covar.transpose(-1, -2)).squeeze(-1),
            )

            # Compute the predictive covariance
            is_root_lt = isinstance(variational_dist.lazy_covariance_matrix, RootLazyTensor)
            is_repeated_root_lt = isinstance(
                variational_dist.lazy_covariance_matrix, BatchRepeatLazyTensor
            ) and isinstance(variational_dist.lazy_covariance_matrix.base_lazy_tensor, RootLazyTensor)
            if is_root_lt:
                predictive_covar = RootLazyTensor(
                    induc_data_covar.transpose(-1, -2) @ variational_dist.lazy_covariance_matrix.root.evaluate()
                )
            elif is_repeated_root_lt:
                predictive_covar = RootLazyTensor(
                    induc_data_covar.transpose(-1, -2)
                    @ variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate()
                )
            else:
                predictive_covar = MatmulLazyTensor(
                    induc_data_covar.transpose(-1, -2), predictive_covar @ induc_data_covar
                )

            if self.training:
                data_covariance = DiagLazyTensor((data_data_covar.diag() - interp_data_data_var).clamp(0, math.inf))
            else:
                neg_induc_data_data_covar = torch.matmul(
                    induc_data_covar.transpose(-1, -2).mul(-1), induc_induc_covar.inv_matmul(induc_data_covar)
                )
                data_covariance = data_data_covar + neg_induc_data_data_covar
            predictive_covar = PsdSumLazyTensor(predictive_covar, data_covariance)

            # Save the logdet, mean_diff_inv_quad, prior distribution for the ELBO
            if self.training:
                prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)
                add_to_cache(self, "prior_distribution_memo", prior_dist)
                add_to_cache(self, "logdet_memo", -logdet)
                add_to_cache(self, "mean_diff_inv_quad_memo", mean_diff_inv_quad)

            return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x, prior=False):
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x)

        # Delete previously cached items from the training distribution
        if self.training:
            clear_cache_hook(self)
        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            self._variational_distribution.initialize_variational_distribution(prior_dist)
            self.variational_params_initialized.fill_(1)

        return Module.__call__(self, x)
