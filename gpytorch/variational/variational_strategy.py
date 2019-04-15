#!/usr/bin/env python3

import math
import torch
from .. import settings
from ..lazy import DiagLazyTensor, CachedCGLazyTensor, CholLazyTensor, PsdSumLazyTensor, RootLazyTensor
from ..module import Module
from ..distributions import MultivariateNormal
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.cholesky import psd_safe_cholesky
from ..utils.memoize import cached


class VariationalStrategy(Module):
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
    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=False):
        """
        Args:
            model (:obj:`gpytorch.model.AbstractVariationalGP`): Model this strategy is applied to. Typically passed in
            when the VariationalStrategy is created in the __init__ method of the user defined model.
            inducing_points (torch.tensor): Tensor containing a set of inducing points to use for variational inference.
            variational_distribution (:obj:`gpytorch.variational.VariationalDistribution`): A VariationalDistribution
                object that represents the form of the variational distribution :math:`q(u)`
            learn_inducing_locations (bool): Whether or not the inducing point locations should be learned (e.g. SVGP).
        """
        super(VariationalStrategy, self).__init__()
        object.__setattr__(self, "model", model)

        inducing_points = inducing_points.clone()

        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        if learn_inducing_locations:
            self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("inducing_points", inducing_points)

        self.variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.
        """
        out = self.model.forward(self.inducing_points)
        res = MultivariateNormal(
            out.mean, out.lazy_covariance_matrix.add_jitter()
        )
        return res

    def kl_divergence(self):
        variational_dist_u = self.variational_distribution.variational_distribution
        prior_dist = self.prior_distribution

        kl_divergence = torch.distributions.kl.kl_divergence(variational_dist_u, prior_dist)
        return kl_divergence

    def initialize_variational_dist(self):
        """
        Describes what distribution to pass to the VariationalDistribution to initialize with. Most commonly, this
        should be the prior distribution for the inducing points, N(m_u, K_uu). However, if a subclass assumes
        a different parameterization of the variational distribution, it may need to modify what the prior is
        with respect to that reparameterization.
        """
        prior_dist = self.prior_distribution
        eval_prior_dist = torch.distributions.MultivariateNormal(
            loc=prior_dist.mean,
            scale_tril=psd_safe_cholesky(prior_dist.covariance_matrix),
        )
        self.variational_distribution.initialize_variational_distribution(eval_prior_dist)

    def forward(self, x):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        Args:
            x (torch.tensor): Locations x to get the variational posterior of the function values at.
        Returns:
            :obj:`gpytorch.distributions.MultivariateNormal`: The distribution q(f|x)
        """
        variational_dist = self.variational_distribution.variational_distribution
        inducing_points = self.inducing_points
        if inducing_points.shape[:-2] < x.shape[:-2]:
            batch_shape = _mul_broadcast_shape(inducing_points.shape[:-2], x.shape[:-2])
            inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
            x = x.expand(*batch_shape, *x.shape[-2:])
            variational_dist = variational_dist.expand(batch_shape)

        # If our points equal the inducing points, we're done
        if torch.equal(x, inducing_points):
            return variational_dist

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
            root_variational_covar = variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate()

            # If we're less than a certain size, we'll compute the Cholesky decomposition of induc_induc_covar
            cholesky = False
            if settings.fast_computations.log_prob.off() or (num_induc <= settings.max_cholesky_size.value()):
                induc_induc_covar = CholLazyTensor(induc_induc_covar.cholesky())
                cholesky = True

            # Cache the CG results
            # For now: run variational inference without a preconditioner
            # The preconditioner screws things up for some reason
            with settings.max_preconditioner_size(0):
                # Cache the CG results
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

            # Compute predictive mean/covariance
            inv_products = induc_induc_covar.inv_matmul(induc_data_covar, left_tensors.transpose(-1, -2))
            predictive_mean = torch.add(test_mean, inv_products[..., 0, :])
            predictive_covar = RootLazyTensor(inv_products[..., 1:, :].transpose(-1, -2))
            if self.training:
                interp_data_data_var, _ = induc_induc_covar.inv_quad_logdet(
                    induc_data_covar, logdet=False, reduce_inv_quad=False
                )
                data_covariance = DiagLazyTensor((data_data_covar.diag() - interp_data_data_var).clamp(0, math.inf))
            else:
                neg_induc_data_data_covar = induc_induc_covar.inv_matmul(
                    induc_data_covar, left_tensor=induc_data_covar.transpose(-1, -2).mul(-1)
                )
                data_covariance = data_data_covar + neg_induc_data_data_covar
            predictive_covar = PsdSumLazyTensor(predictive_covar, data_covariance)

            return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x):
        if not self.variational_params_initialized.item():
            self.initialize_variational_dist()
            self.variational_params_initialized.fill_(1)
        if self.training:
            if hasattr(self, "_memoize_cache"):
                delattr(self, "_memoize_cache")
                self._memoize_cache = dict()

        return super(VariationalStrategy, self).__call__(x)
