#!/usr/bin/env python3

import math
import torch
from .. import beta_features
from ..lazy import RootLazyTensor, PsdSumLazyTensor, DiagLazyTensor
from ..module import Module
from ..distributions import MultivariateNormal


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
    def prior_distribution(self):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.
        """
        if hasattr(self, "_prior_distribution_memo"):
            return self._prior_distribution_memo
        else:
            out = self.model.forward(self.inducing_points)
            return MultivariateNormal(out.mean, out.lazy_covariance_matrix.evaluate_kernel().add_jitter())

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
        if torch.equal(x, self.inducing_points):
            return variational_dist
        else:
            n_induc = self.inducing_points.size(-2)
            full_inputs = torch.cat([self.inducing_points, x], dim=-2)
            full_output = self.model.forward(full_inputs)
            full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

            test_mean = full_mean[..., n_induc:]
            induc_mean = full_mean[..., :n_induc]
            induc_induc_covar = full_covar[..., :n_induc, :n_induc].add_jitter()
            induc_data_covar = full_covar[..., :n_induc, n_induc:]
            data_data_covar = full_covar[..., n_induc:, n_induc:]
            var_dist_mean = variational_dist.mean

            # Cache the prior distribution, for faster training
            if self.training:
                prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)
                self._prior_distribution_memo = prior_dist

            # Compute predictive mean/covariance
            induc_data_covar = induc_data_covar.evaluate()
            inv_product = induc_induc_covar.inv_matmul(induc_data_covar)
            factor = variational_dist.lazy_covariance_matrix.root_decomposition().root.matmul(inv_product)
            predictive_mean = torch.add(
                test_mean, inv_product.transpose(-1, -2).matmul((var_dist_mean - induc_mean).unsqueeze(-1)).squeeze(-1)
            )
            predictive_covar = RootLazyTensor(factor.transpose(-2, -1))

            if beta_features.diagonal_correction.on():
                fake_diagonal = (inv_product * induc_data_covar).sum(-2)
                real_diagonal = data_data_covar.diag()
                diag_correction = DiagLazyTensor((real_diagonal - fake_diagonal).clamp(0, math.inf))
                predictive_covar = PsdSumLazyTensor(predictive_covar, diag_correction)

            return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x):
        if not self.variational_params_initialized.item():
            self.variational_distribution.initialize_variational_distribution(self.prior_distribution)
            self.variational_params_initialized.fill_(1)
        return super(VariationalStrategy, self).__call__(x)
