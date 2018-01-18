import logging
import gpytorch
import torch
from torch.autograd import Variable
from ..module import Module
from ..random_variables import GaussianRandomVariable
from ..likelihoods import GaussianLikelihood
from ..lazy import LazyVariable, CholLazyVariable, InterpolatedLazyVariable, MatmulLazyVariable, NonLazyVariable
from ..utils import StochasticLQ, left_interp


class ExactGP(Module):
    def __init__(self, train_inputs, train_targets, likelihood):
        if torch.is_tensor(train_inputs):
            train_inputs = train_inputs,
        if not all(torch.is_tensor(train_input) for train_input in train_inputs):
            raise RuntimeError('Train inputs must be a tensor, or a list/tuple of tensors')

        super(ExactGP, self).__init__()
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.likelihood = likelihood

        self.has_computed_alpha = False
        self.has_computed_low_rank = False

    def _apply(self, fn):
        self.train_inputs = tuple(fn(train_input) for train_input in self.train_inputs)
        self.train_targets = fn(self.train_targets)
        return super(ExactGP, self)._apply(fn)

    def marginal_log_likelihood(self, likelihood, output, target, n_data=None):
        """
        A special MLL designed for exact inference

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - output: (GaussianRandomVariable) - the output of the GP model
        - target: (Variable) - target
        """
        if not isinstance(likelihood, GaussianLikelihood):
            raise RuntimeError('Likelihood must be Gaussian for exact inference')

        if not torch.equal(target.data, self.train_targets):
            raise RuntimeError('You must train on the training targets!')

        mean, covar = likelihood(output).representation()
        n_data = len(target)
        return gpytorch.exact_gp_marginal_log_likelihood(covar, target - mean).div(n_data)

    def train(self, mode=True):
        if mode:
            self.has_computed_alpha = False
            self.has_computed_low_rank = False
        return super(ExactGP, self).train(mode)

    def __call__(self, *args, **kwargs):
        train_inputs = tuple(Variable(train_input) for train_input in self.train_inputs)

        # Training mode: optimizing
        if self.training:
            if not all([torch.equal(train_input, input) for train_input, input in zip(train_inputs, args)]):
                raise RuntimeError('You must train on the training inputs!')
            return super(ExactGP, self).__call__(*args, **kwargs)

        # Posterior mode
        else:
            if all([torch.equal(train_input, input) for train_input, input in zip(train_inputs, args)]):
                logging.warning('The input matches the stored training data. '
                                'Did you forget to call model.train()?')

            # Exact inference
            n_train = train_inputs[0].size(0)
            full_inputs = tuple(torch.cat([train_input, input]) for train_input, input in zip(train_inputs, args))
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if not isinstance(full_output, GaussianRandomVariable):
                raise RuntimeError('ExactGP.forward must return a GaussianRandomVariable')
            full_mean, full_covar = full_output.representation()

            train_mean = full_mean[:n_train]
            test_mean = full_mean[n_train:]
            train_train_covar = gpytorch.add_diag(full_covar[:n_train, :n_train], self.likelihood.log_noise.exp())
            train_test_covar = full_covar[:n_train, n_train:]
            test_train_covar = full_covar[n_train:, :n_train]
            test_test_covar = full_covar[n_train:, n_train:]

            # Calculate alpha cache
            if not self.has_computed_alpha:
                train_residual = Variable(self.train_targets) - train_mean
                alpha = gpytorch.inv_matmul(train_train_covar, train_residual)
                if isinstance(full_covar, InterpolatedLazyVariable):
                    right_interp = InterpolatedLazyVariable(test_train_covar.base_lazy_variable,
                                                            left_interp_indices=None, left_interp_values=None,
                                                            right_interp_indices=test_train_covar.right_interp_indices,
                                                            right_interp_values=test_train_covar.right_interp_values)
                    alpha = right_interp.matmul(alpha)

                self.alpha = alpha
                self.has_computed_alpha = True

            # Calculate low rank cache, if necessary
            if not self.has_computed_low_rank and gpytorch.functions.fast_pred_var:
                if isinstance(train_train_covar, LazyVariable):
                    train_train_representation = [var.data for var in train_train_covar.representation()]
                    train_train_matmul = train_train_covar._matmul_closure_factory(*train_train_representation)
                    tensor_cls = train_train_covar.tensor_cls
                else:
                    train_train_matmul = train_train_covar.data.matmul
                    tensor_cls = type(train_train_covar.data)
                max_iter = min(n_train, gpytorch.functions.max_lanczos_iterations)
                lq_object = StochasticLQ(cls=tensor_cls, max_iter=max_iter)
                init_vector = tensor_cls(n_train, 1).normal_()
                init_vector /= torch.norm(init_vector, 2, 0)
                q_mat, t_mat = lq_object.lanczos_batch(train_train_matmul, init_vector)
                self.low_rank_left = Variable(q_mat[0].matmul(t_mat[0].inverse()))
                self.low_rank_right = Variable(q_mat[0].transpose(-1, -2))
                self.has_computed_low_rank = True

            # Calculate mean
            if isinstance(full_covar, InterpolatedLazyVariable):
                left_interp_indices = test_train_covar.left_interp_indices
                left_interp_values = test_train_covar.left_interp_values
                predictive_mean = left_interp(left_interp_indices, left_interp_values, self.alpha) + test_mean
            elif isinstance(test_train_covar, LazyVariable):
                predictive_mean = test_train_covar.matmul(self.alpha) + test_mean
            else:
                predictive_mean = torch.addmv(test_mean, test_train_covar, self.alpha)

            # Calculate covar
            if gpytorch.functions.fast_pred_var:
                if not isinstance(test_test_covar, LazyVariable):
                    test_test_covar = NonLazyVariable(test_test_covar)
                covar_correction_left = test_train_covar.matmul(self.low_rank_left)
                covar_correction_right = test_train_covar.matmul(self.low_rank_right.transpose(-1, -2))
                covar_correction_right = covar_correction_right.transpose(-1, -2)
                covar_correction = MatmulLazyVariable(covar_correction_left, covar_correction_right).mul(-1)
                predictive_covar = test_test_covar + covar_correction
            else:
                if isinstance(train_test_covar, LazyVariable):
                    train_test_covar = train_test_covar.evaluate()
                if isinstance(test_train_covar, LazyVariable):
                    test_train_covar = train_test_covar.t()
                if not isinstance(test_test_covar, LazyVariable):
                    test_test_covar = NonLazyVariable(test_test_covar)
                covar_correction_rhs = gpytorch.inv_matmul(train_train_covar, train_test_covar).mul_(-1)
                predictive_covar = test_test_covar + MatmulLazyVariable(test_train_covar, covar_correction_rhs)
            return GaussianRandomVariable(predictive_mean, predictive_covar)
