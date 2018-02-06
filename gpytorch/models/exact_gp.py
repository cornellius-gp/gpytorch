import logging
import gpytorch
import torch
from torch.autograd import Variable
from ..module import Module
from ..random_variables import GaussianRandomVariable
from ..likelihoods import GaussianLikelihood


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

        self.mean_cache = None
        self.covar_cache = None

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
            self.mean_cache = None
            self.covar_cache = None
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
            full_inputs = tuple(torch.cat([train_input, input]) for train_input, input in zip(train_inputs, args))
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if not isinstance(full_output, GaussianRandomVariable):
                raise RuntimeError('ExactGP.forward must return a GaussianRandomVariable')
            full_mean, full_covar = full_output.representation()

            noise = self.likelihood.log_noise.exp()
            predictive_mean, mean_cache = gpytorch.exact_predictive_mean(full_covar, full_mean,
                                                                         Variable(self.train_targets),
                                                                         noise, self.mean_cache)
            predictive_covar, covar_cache = gpytorch.exact_predictive_covar(full_covar, self.train_targets.size(-1),
                                                                            noise, self.covar_cache)

            self.mean_cache = mean_cache
            self.covar_cache = covar_cache
            return GaussianRandomVariable(predictive_mean, predictive_covar)
