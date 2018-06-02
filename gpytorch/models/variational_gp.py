from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Variable
from .. import beta_features
from ..functions import inv_matmul
from ..random_variables import GaussianRandomVariable
from ..lazy import LazyVariable, RootLazyVariable, MatmulLazyVariable, NonLazyVariable
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
            if not torch.equal(inputs.data, self.inducing_points):
                raise RuntimeError("You must train on the training inputs!")

            prior_output = self.prior_output()
            # Initialize variational parameters, if necessary
            if not self.variational_params_initialized[0]:
                mean_init = prior_output.mean().data
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
            full_inputs = torch.cat([Variable(self.inducing_points), inputs])
            full_output = super(VariationalGP, self).__call__(full_inputs)
            full_mean, full_covar = full_output.representation()

            induc_mean = full_mean[:n_induc]
            test_mean = full_mean[n_induc:]
            induc_induc_covar = full_covar[:n_induc, :n_induc]
            induc_test_covar = full_covar[:n_induc, n_induc:]
            test_induc_covar = full_covar[n_induc:, :n_induc]
            test_test_covar = full_covar[n_induc:, n_induc:]

            # Compute alpha cache
            if not self.has_computed_alpha:
                self.alpha = inv_matmul(induc_induc_covar, variational_output.mean() - induc_mean)
                self.has_computed_alpha = True

            # Compute chol cache, if necessary
            if not self.has_computed_root and beta_features.fast_pred_var.on():
                if not isinstance(induc_induc_covar, LazyVariable):
                    induc_induc_covar = NonLazyVariable(induc_induc_covar)
                self.prior_root_inv = induc_induc_covar.root_inv_decomposition()

                chol_variational_output = variational_output.covar().root.evaluate()
                self.variational_root = inv_matmul(induc_induc_covar, chol_variational_output)
                self.has_computed_root = True

            # Test mean
            predictive_mean = torch.add(test_mean, test_induc_covar.matmul(self.alpha))

            # Test covariance
            if not isinstance(test_test_covar, LazyVariable):
                predictive_covar = NonLazyVariable(test_test_covar)
            else:
                predictive_covar = test_test_covar
            if beta_features.fast_pred_var.on():
                correction = RootLazyVariable(test_induc_covar.matmul(self.prior_root_inv)).mul(-1)
                correction = correction + RootLazyVariable(test_induc_covar.matmul(self.variational_root))
                predictive_covar = predictive_covar + correction
            else:
                if isinstance(induc_test_covar, LazyVariable):
                    induc_test_covar = induc_test_covar.evaluate()
                inv_product = inv_matmul(induc_induc_covar, induc_test_covar)
                factor = variational_output.covar().root_decomposition().matmul(inv_product)
                right_factor = factor - inv_product
                left_factor = (factor - induc_test_covar).transpose(-1, -2)
                predictive_covar = predictive_covar + MatmulLazyVariable(left_factor, right_factor)

            output = GaussianRandomVariable(predictive_mean, predictive_covar)
            return output
