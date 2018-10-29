from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from ..lazy import RootLazyTensor, MatmulLazyTensor
from .. import beta_features
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

        if learn_inducing_locations:
            self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("inducing_points", inducing_points)

        self.variational_distribution = variational_distribution

        self.has_computed_alpha = False
        self.has_computed_root = False

    def train(self, mode=True):
        if mode:
            self.has_computed_alpha = False
            self.has_computed_root = False
        return super(VariationalStrategy, self).train(mode)

    @property
    def prior_distribution(self):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.
        """
        out = self.model.forward(self.inducing_points)
        return MultivariateNormal(out.mean, out.lazy_covariance_matrix.evaluate_kernel())

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
            n_induc = len(self.inducing_points)
            full_inputs = torch.cat([self.inducing_points, x])
            full_output = self.model.forward(full_inputs)
            full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

            test_mean = full_mean[n_induc:]
            induc_mean = full_mean[:n_induc]
            induc_induc_covar = full_covar[:n_induc, :n_induc]
            induc_data_covar = full_covar[:n_induc, n_induc:]
            data_induc_covar = full_covar[n_induc:, :n_induc]
            data_data_covar = full_covar[n_induc:, n_induc:]

            # Compute alpha cache
            if not self.has_computed_alpha:
                self.alpha = induc_induc_covar.inv_matmul(variational_dist.mean - induc_mean)
                # Don't consider this computation a cache if we are in training mode.
                if not self.training:
                    self.has_computed_alpha = True

            # Compute chol cache, if necessary
            if not self.training and not self.has_computed_root and beta_features.fast_pred_var.on():
                self.prior_root_inv = induc_induc_covar.root_inv_decomposition()

                chol_variational_output = variational_dist.lazy_covariance_matrix.root.evaluate()
                self.variational_root = induc_induc_covar.inv_matmul(chol_variational_output)
                self.has_computed_root = True

            # Compute predictive mean
            predictive_mean = torch.add(test_mean, data_induc_covar.matmul(self.alpha))

            # Compute predictive covariance
            predictive_covar = data_data_covar
            if beta_features.fast_pred_var.on():
                correction = RootLazyTensor(data_induc_covar.matmul(self.prior_root_inv)).mul(-1)
                correction = correction + RootLazyTensor(data_induc_covar.matmul(self.variational_root))
                predictive_covar = predictive_covar + correction
            else:
                induc_data_covar = induc_data_covar.evaluate()
                inv_product = induc_induc_covar.inv_matmul(induc_data_covar)
                factor = variational_dist.lazy_covariance_matrix.root_decomposition().matmul(inv_product)
                right_factor = factor - inv_product
                left_factor = (factor - induc_data_covar).transpose(-1, -2)
                predictive_covar = predictive_covar + MatmulLazyTensor(left_factor, right_factor)

            return MultivariateNormal(predictive_mean, predictive_covar)
