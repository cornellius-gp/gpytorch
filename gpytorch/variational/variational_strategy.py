#!/usr/bin/env python3

import math
import torch
from .. import beta_features, settings
from ..lazy import DiagLazyTensor, CachedCGLazyTensor, PsdSumLazyTensor, RootLazyTensor
from ..module import Module
from ..distributions import MultivariateNormal
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
            out.mean, out.lazy_covariance_matrix.evaluate_kernel().add_jitter()
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
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            eval_prior_dist = torch.distributions.MultivariateNormal(
                loc=prior_dist.mean,
                covariance_matrix=prior_dist.covariance_matrix
            )
            self.variational_distribution.initialize_variational_distribution(eval_prior_dist)
            self.variational_params_initialized.fill_(1)

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
        if inducing_points.dim() < x.dim():
            inducing_points = inducing_points.expand(*x.shape[:-2], *inducing_points.shape[-2:])
            variational_dist = variational_dist.expand(x.shape[:-2])

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
            induc_induc_covar = full_covar[..., :num_induc, :num_induc].evaluate_kernel().add_jitter()
            induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
            data_data_covar = full_covar[..., num_induc:, num_induc:]
            root_variational_covar = variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate()

            # Cache the CG results
            # For now: run variational inference without a preconditioner
            # The preconditioner screws things up for some reason
            with settings.max_preconditioner_size(0):
                # Cache the CG results
                left_tensors = torch.cat([mean_diff, root_variational_covar], -1)
                with torch.no_grad():
                    eager_rhs = torch.cat([left_tensors, induc_data_covar], -1)
                    solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLazyTensor.precompute_terms(
                        induc_induc_covar, eager_rhs.detach(), logdet_terms=self.training,
                        include_tmats=(not settings.skip_logdet_forward.on())
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
                        eager_rhss.append(torch.cat([probe_vecs, induc_data_covar], -1))
                        solves.append(torch.cat([probe_vec_solves, solve[..., left_tensors.size(-1):]], -1))
                induc_induc_covar = CachedCGLazyTensor(
                    induc_induc_covar, eager_rhss=eager_rhss, solves=solves, probe_vectors=probe_vecs,
                    probe_vector_norms=probe_vec_norms, probe_vector_solves=probe_vec_solves,
                    probe_vector_tmats=tmats,
                )

            # Compute predictive mean/covariance
            inv_products = induc_induc_covar.inv_matmul(induc_data_covar, left_tensors.transpose(-1, -2))
            predictive_mean = torch.add(test_mean, inv_products[..., 0, :])
            predictive_covar = RootLazyTensor(inv_products[..., 1:, :].transpose(-1, -2))
            if beta_features.diagonal_correction.on():
                interp_data_data_var, _ = induc_induc_covar.inv_quad_logdet(
                    induc_data_covar, logdet=False, reduce_inv_quad=False
                )
                diag_correction = DiagLazyTensor((data_data_covar.diag() - interp_data_data_var).clamp(0, math.inf))
                predictive_covar = PsdSumLazyTensor(predictive_covar, diag_correction)

            # Save the logdet, mean_diff_inv_quad, prior distribution for the ELBO
            if self.training:
                self._memoize_cache["prior_distribution_memo"] = MultivariateNormal(induc_mean, induc_induc_covar)

            return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x):
        self.initialize_variational_dist()
        if self.training:
            if hasattr(self, "_memoize_cache"):
                delattr(self, "_memoize_cache")
                self._memoize_cache = dict()

        return super(VariationalStrategy, self).__call__(x)
