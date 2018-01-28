import logging
import gpytorch
import torch
from torch.autograd import Variable
from .. import beta_features
from ..module import Module
from ..random_variables import GaussianRandomVariable
from ..likelihoods import GaussianLikelihood
from ..lazy import LazyVariable, InterpolatedLazyVariable, MatmulLazyVariable, NonLazyVariable, RootLazyVariable
from ..utils import left_interp


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
        self.has_computed_root_inv = False

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
            self.has_computed_root_inv = False
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
                    # We can get a better alpha cache with InterpolatedLazyVariables (Kiss-GP)
                    # This allows for constant time predictions
                    right_interp = InterpolatedLazyVariable(test_train_covar.base_lazy_variable,
                                                            left_interp_indices=None, left_interp_values=None,
                                                            right_interp_indices=test_train_covar.right_interp_indices,
                                                            right_interp_values=test_train_covar.right_interp_values)
                    alpha = right_interp.matmul(alpha)

                self.alpha = alpha
                self.has_computed_alpha = True

            # Calculate root inverse cache, if necessary
            # This enables fast predictive variances
            if not self.has_computed_root_inv and beta_features.fast_pred_var.on():
                if not isinstance(train_train_covar, LazyVariable):
                    train_train_covar = NonLazyVariable(train_train_covar)
                root_inv = train_train_covar.root_inv_decomposition().root.evaluate()
                if isinstance(full_covar, InterpolatedLazyVariable):
                    # We can get a better root_inv cache with InterpolatedLazyVariables (Kiss-GP)
                    # This allows for constant time predictive variances
                    right_interp = InterpolatedLazyVariable(test_train_covar.base_lazy_variable,
                                                            left_interp_indices=None, left_interp_values=None,
                                                            right_interp_indices=test_train_covar.right_interp_indices,
                                                            right_interp_values=test_train_covar.right_interp_values)
                    root_inv = right_interp.matmul(root_inv)

                self.root_inv = root_inv
                self.has_computed_root_inv = True

            # Calculate mean
            if isinstance(full_covar, InterpolatedLazyVariable):
                # Constant time predictions with InterpolatedLazyVariables (Kiss-GP)
                left_interp_indices = test_train_covar.left_interp_indices
                left_interp_values = test_train_covar.left_interp_values
                predictive_mean = left_interp(left_interp_indices, left_interp_values, self.alpha) + test_mean
            else:
                # O(n) predictions with normal LazyVariables
                predictive_mean = test_train_covar.matmul(self.alpha) + test_mean

            # Calculate covar
            if beta_features.fast_pred_var.on():
                # Compute low-rank approximation of covariance matrix - much faster!
                if not isinstance(test_test_covar, LazyVariable):
                    test_test_covar = NonLazyVariable(test_test_covar)

                if isinstance(full_covar, InterpolatedLazyVariable):
                    # Constant time predictive var with InterpolatedLazyVariables (Kiss-GP)
                    left_interp_indices = test_train_covar.left_interp_indices
                    left_interp_values = test_train_covar.left_interp_values
                    covar_correction_root = left_interp(left_interp_indices, left_interp_values, self.root_inv)
                    predictive_covar = test_test_covar + RootLazyVariable(covar_correction_root).mul(-1)
                else:
                    # O(n) predictions with normal LazyVariables
                    covar_correction_root = test_train_covar.matmul(self.root_inv)
                    covar_correction = RootLazyVariable(covar_correction_root).mul(-1)
                    predictive_covar = test_test_covar + covar_correction
            else:
                # Compute full covariance matrix - much slower
                if isinstance(train_test_covar, LazyVariable):
                    train_test_covar = train_test_covar.evaluate()
                if isinstance(test_train_covar, LazyVariable):
                    test_train_covar = train_test_covar.t()
                if not isinstance(test_test_covar, LazyVariable):
                    test_test_covar = NonLazyVariable(test_test_covar)
                covar_correction_rhs = gpytorch.inv_matmul(train_train_covar, train_test_covar).mul(-1)
                predictive_covar = test_test_covar + MatmulLazyVariable(test_train_covar, covar_correction_rhs)
            return GaussianRandomVariable(predictive_mean, predictive_covar)
