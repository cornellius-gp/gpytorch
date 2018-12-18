#!/usr/bin/env python3

import math
import torch
from .. import beta_features
from ..lazy import DiagLazyTensor, NonLazyTensor, PsdSumLazyTensor, RootLazyTensor
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
            num_induc = self.inducing_points.size(-2)
            full_inputs = torch.cat([self.inducing_points, x], dim=-2)
            full_output = self.model.forward(full_inputs)
            full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

            # Mean terms
            test_mean = full_mean[..., num_induc:]
            induc_mean = full_mean[..., :num_induc]
            var_dist_mean = variational_dist.mean
            mean_diff = (var_dist_mean - induc_mean).unsqueeze(-1)

            # Covariance terms
            induc_induc_covar = NonLazyTensor(full_covar[..., :num_induc, :num_induc].evaluate()).add_jitter()
            # TODO: FIX
            induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
            data_data_covar = full_covar[..., num_induc:, num_induc:]
            root_variational_covar = variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate()

            # Compute all products with `induc_induc_covar^{-1}` simultaneously
            eager_rhs = torch.cat([mean_diff, induc_data_covar, root_variational_covar.transpose(-1, -2)], -1)
            inv_products = induc_induc_covar.inv_matmul(induc_data_covar, eager_rhs.transpose(-1, -2))

            # Cache the prior distribution, for faster training
            if self.training:
                prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)
                self._prior_distribution_memo = prior_dist

            # Compute predictive mean/covariance
            predictive_mean = torch.add(test_mean, inv_products[..., 0, :])
            predictive_covar = RootLazyTensor(inv_products[..., -num_induc:, :].transpose(-1, -2))

            # Compute a diagonal correction for the covariance. It is the diagonal of the Schur complement
            # K_{data,data} - K_{data,induc} K_{induc,induc}^{-1} K_{induc,data}
            if beta_features.diagonal_correction.on():
                # interp_data_data_covar = K_{data,induc} K_{induc,induc}^{-1} K_{induc,data}
                interp_data_data_covar = inv_products[..., 1:-num_induc, :]
                diag_correction = DiagLazyTensor(
                    (data_data_covar.diag() - interp_data_data_covar.diagonal(dim1=-2, dim2=-1)).clamp(0, math.inf)
                )
                predictive_covar = PsdSumLazyTensor(predictive_covar, diag_correction)

            return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x):
        if not self.variational_params_initialized.item():
            self.variational_distribution.initialize_variational_distribution(self.prior_distribution)
            self.variational_params_initialized.fill_(1)
        return super(VariationalStrategy, self).__call__(x)
