import gpytorch
import torch
from torch.autograd import Variable
from ..random_variables import GaussianRandomVariable
from ..lazy import LazyVariable, CholLazyVariable, MatmulLazyVariable, NonLazyVariable
from ..variational import MVNVariationalStrategy
from .abstract_variational_gp import AbstractVariationalGP
from ..utils import StochasticLQ


class VariationalGP(AbstractVariationalGP):
    def __init__(self, train_input):
        if not torch.is_tensor(train_input):
            raise RuntimeError('VariationalGP must take a single tensor train_input')
        super(VariationalGP, self).__init__(train_input)

        self.has_computed_alpha = False
        self.has_computed_chol = False

    def train(self, mode=True):
        if mode:
            self.has_computed_alpha = False
            self.has_computed_chol = False
        return super(VariationalGP, self).train(mode)

    def __call__(self, inputs, **kwargs):
        # Training mode: optimizing
        if self.training:
            if not torch.equal(inputs.data, self.inducing_points):
                raise RuntimeError('You must train on the training inputs!')

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
            self.update_variational_strategy('inducing_point_strategy', new_variational_strategy)
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
                self.alpha = gpytorch.inv_matmul(induc_induc_covar, variational_output.mean() - induc_mean)
                self.has_computed_alpha = True

            # Compute chol cache, if necessary
            if not self.has_computed_chol and gpytorch.functions.fast_pred_var:
                if isinstance(induc_induc_covar, LazyVariable):
                    induc_induc_representation = [var.data for var in induc_induc_covar.representation()]
                    induc_induc_matmul = induc_induc_covar._matmul_closure_factory(*induc_induc_representation)
                    tensor_cls = induc_induc_covar.tensor_cls
                else:
                    induc_induc_matmul = induc_induc_covar.data.matmul
                    tensor_cls = type(induc_induc_covar.data)
                max_iter = min(n_induc, gpytorch.functions.max_lanczos_iterations)
                lq_object = StochasticLQ(cls=tensor_cls, max_iter=max_iter)
                init_vector = tensor_cls(n_induc, 1).normal_()
                init_vector /= torch.norm(init_vector, 2, 0)
                q_mat, t_mat = lq_object.lanczos_batch(induc_induc_matmul, init_vector)
                self.prior_chol = Variable(q_mat[0].matmul(t_mat[0].potrf().inverse()))

                self.variational_chol = gpytorch.inv_matmul(induc_induc_covar, variational_output.covar().lhs)
                self.has_computed_chol = True

            # Test mean
            predictive_mean = torch.add(test_mean, test_induc_covar.matmul(self.alpha))

            # Test covariance
            if not isinstance(test_test_covar, LazyVariable):
                predictive_covar = NonLazyVariable(test_test_covar)
            else:
                predictive_covar = test_test_covar
            if gpytorch.functions.fast_pred_var:
                predictive_covar = predictive_covar + CholLazyVariable(test_induc_covar.matmul(self.prior_chol)).mul(-1)
                predictive_covar = predictive_covar + CholLazyVariable(test_induc_covar.matmul(self.variational_chol))
            else:
                if isinstance(induc_test_covar, LazyVariable):
                    induc_test_covar = induc_test_covar.evaluate()
                inv_product = gpytorch.inv_matmul(induc_induc_covar, induc_test_covar)
                factor = variational_output.covar().chol_matmul(inv_product)
                right_factor = factor - inv_product
                left_factor = (factor - induc_test_covar).transpose(-1, -2)
                predictive_covar = predictive_covar + MatmulLazyVariable(left_factor, right_factor)

            output = GaussianRandomVariable(predictive_mean, predictive_covar)
            return output
