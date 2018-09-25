from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .. import beta_features
from ..functions import inv_matmul
from ..distributions import MultivariateNormal
from ..lazy import RootLazyTensor, MatmulLazyTensor
from ..variational import MVNVariationalStrategy
from .abstract_variational_gp import AbstractVariationalGP


class VariationalGP(AbstractVariationalGP):
    def __init__(self, train_input):
        if not torch.is_tensor(train_input):
            raise RuntimeError("VariationalGP must take a single tensor train_input")
        super(VariationalGP, self).__init__(train_input)

        self.has_computed_alpha = False
        self.has_computed_root = False

    def train(self, mode=True):
        if mode:
            self.has_computed_alpha = False
            self.has_computed_root = False
        return super(VariationalGP, self).train(mode)

    def __call__(self, inputs, **kwargs):
        # Training mode: optimizing
        if self.training:
            if not torch.equal(inputs, self.inducing_points):
                raise RuntimeError("You must train on the training inputs!")

            prior_output = self.prior_output()
            # Initialize variational parameters, if necessary
            if not self.variational_params_initialized.item():
                mean_init = prior_output.mean
                chol_covar_init = torch.eye(len(mean_init)).type_as(mean_init)
                self.variational_mean.data.copy_(mean_init)
                self.chol_variational_covar.data.copy_(chol_covar_init)
                self.variational_params_initialized.fill_(1)

            variational_output = self.variational_output()
            new_variational_strategy = MVNVariationalStrategy(variational_output, prior_output)
            self.update_variational_strategy("inducing_point_strategy", new_variational_strategy)
            return variational_output

        # Posterior mode
        else:
            variational_output = self.variational_output()

            n_induc = len(self.inducing_points)
            full_inputs = torch.cat([self.inducing_points, inputs])
            full_output = super(VariationalGP, self).__call__(full_inputs)
            full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

            induc_mean = full_mean[:n_induc]
            test_mean = full_mean[n_induc:]
            induc_induc_covar = full_covar[:n_induc, :n_induc]
            induc_test_covar = full_covar[:n_induc, n_induc:]
            test_induc_covar = full_covar[n_induc:, :n_induc]
            test_test_covar = full_covar[n_induc:, n_induc:]

            # Compute alpha cache
            if not self.has_computed_alpha:
                self.alpha = inv_matmul(induc_induc_covar, variational_output.mean - induc_mean)
                self.has_computed_alpha = True

            # Compute chol cache, if necessary
            if not self.has_computed_root and beta_features.fast_pred_var.on():
                self.prior_root_inv = induc_induc_covar.root_inv_decomposition()

                chol_variational_output = variational_output.lazy_covariance_matrix.root.evaluate()
                self.variational_root = inv_matmul(induc_induc_covar, chol_variational_output)
                self.has_computed_root = True

            # Test mean
            predictive_mean = torch.add(test_mean, test_induc_covar.matmul(self.alpha))

            # Test covariance
            predictive_covar = test_test_covar
            if beta_features.fast_pred_var.on():
                correction = RootLazyTensor(test_induc_covar.matmul(self.prior_root_inv)).mul(-1)
                correction = correction + RootLazyTensor(test_induc_covar.matmul(self.variational_root))
                predictive_covar = predictive_covar + correction
            else:
                induc_test_covar = induc_test_covar.evaluate()
                inv_product = inv_matmul(induc_induc_covar, induc_test_covar)
                factor = variational_output.lazy_covariance_matrix.root_decomposition().matmul(inv_product)
                right_factor = factor - inv_product
                left_factor = (factor - induc_test_covar).transpose(-1, -2)
                predictive_covar = predictive_covar + MatmulLazyTensor(left_factor, right_factor)

            output = MultivariateNormal(predictive_mean, predictive_covar)
            return output
