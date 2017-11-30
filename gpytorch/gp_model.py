import logging
import gpytorch
import torch
from torch.autograd import Variable
from .random_variables import RandomVariable, GaussianRandomVariable
from .lazy import LazyVariable
from .likelihoods import GaussianLikelihood


class GPModel(gpytorch.Module):
    def __init__(self, likelihood):
        super(GPModel, self).__init__()
        self._parameter_groups = {}
        self.likelihood = likelihood

        self.register_buffer('has_computed_alpha', torch.ByteTensor([0]))
        self.register_buffer('has_computed_lanczos', torch.ByteTensor([0]))
        self.register_buffer('alpha', torch.Tensor())
        self.register_buffer('lanczos_q_mat', torch.Tensor())
        self.register_buffer('lanczos_t_mat', torch.Tensor())

    def condition(self, train_inputs, train_target, **kwargs):
        """
        Conditions the model on data. After conditioning, the model functions
        in posterior mode rather than prior mode.

        The last input to args represents the target values (ys)
        All other inputs are input values (xs)

        Args: (Variables) inputs to condition on
        reset (bool) - reset variational parameters and alpha cache (default True)
        """
        if (isinstance(train_inputs, Variable) or isinstance(train_inputs, LazyVariable) or
                isinstance(train_inputs, RandomVariable)):
            train_inputs = train_inputs,

        res = super(GPModel, self).condition(train_inputs, train_target, **kwargs)
        return res

    @property
    def exact_inference(self):
        """
        Returns true if the model performs exact inference (vs. approximate inference)
        """
        return isinstance(self.likelihood, GaussianLikelihood)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def marginal_log_likelihood(self, output, target, n_data=None):
        """
        Returns the marginal log likelihood of the data

        Args:
        - output: (GaussianRandomVariable) - the output of the model
        - target: (Variable) - target
        - n_data: (int) - total number of data points in the set (required only for SGD)
        """
        if n_data is None:
            n_data = len(target)
        n_batch = output.mean().size(0)

        # Exact inference
        if self.exact_inference:
            mean, covar = output.representation()
            return gpytorch.exact_gp_marginal_log_likelihood(covar, target - mean).div(n_data)

        # Approximate inference
        else:
            samples = output._variational_strategy.variational_samples(output)
            n_samples = samples.size(1)
            log_likelihood = self.likelihood.log_probability(samples.view(-1),
                                                             target.unsqueeze(1).repeat(1, n_samples).view(-1))
            log_likelihood = log_likelihood.div(n_samples).div(n_batch)
            kl_divergence = output._variational_strategy.mvn_kl_divergence().div(n_data)

            res = log_likelihood - kl_divergence
            return res

    def __call__(self, *args, **kwargs):
        output = None

        # Posterior mode
        if self.posterior:
            train_xs = self.train_inputs
            train_y = self.train_target
            if all([torch.equal(train_x.data, input.data)
                    for train_x, input in zip(train_xs, args)]):
                logging.warning('The input matches the stored training data. '
                                'Did you forget to call model.train()?')

            # Exact inference
            if self.exact_inference:
                n_train = len(train_xs[0])
                full_inputs = [torch.cat([train_x, input]) for train_x, input in zip(train_xs, args)]
                full_output = super(GPModel, self).__call__(*full_inputs, **kwargs)
                full_mean, full_covar = full_output.representation()

                train_mean = full_mean[:n_train]
                test_mean = full_mean[n_train:]
                train_train_covar = gpytorch.add_diag(full_covar[:n_train, :n_train], self.likelihood.log_noise.exp())
                train_test_covar = full_covar[:n_train, n_train:]
                test_train_covar = full_covar[n_train:, :n_train]
                test_test_covar = full_covar[n_train:, n_train:]

                # Calculate posterior components
                if not self.has_computed_alpha[0]:
                    alpha_strategy = gpytorch.posterior_strategy(train_train_covar)
                    alpha = alpha_strategy.exact_posterior_alpha(train_mean, train_y)
                    self.alpha.copy_(alpha.data)
                    self.has_computed_alpha.fill_(1)
                else:
                    alpha = Variable(self.alpha)

                if not self.has_computed_lanczos[0] and gpytorch.functions.fast_pred_var:
                    lanczos_strategy = gpytorch.posterior_strategy(train_train_covar)
                    q_mat, t_mat = lanczos_strategy.exact_posterior_lanczos()
                    self.lanczos_q_mat[:, :q_mat.size(1)].copy_(q_mat)
                    self.lanczos_t_mat[:t_mat.size(0), :t_mat.size(1)].copy_(t_mat)
                    self.has_computed_lanczos.fill_(1)

                mean_strategy = gpytorch.posterior_strategy(test_train_covar)
                test_mean = mean_strategy.exact_posterior_mean(test_mean, alpha)
                if gpytorch.functions.fast_pred_var:
                    covar_strategy = gpytorch.posterior_strategy(full_covar)
                    test_covar = covar_strategy.exact_posterior_covar_fast(Variable(self.lanczos_q_mat),
                                                                           Variable(self.lanczos_t_mat))
                else:
                    covar_strategy = gpytorch.posterior_strategy(train_train_covar)
                    test_covar = covar_strategy.exact_posterior_covar(test_train_covar, train_test_covar,
                                                                      test_test_covar)
                output = GaussianRandomVariable(test_mean, test_covar)

            # Approximate inference
            else:
                output = super(GPModel, self).__call__(*args, **kwargs)

        # Training or Prior mode
        else:
            output = super(GPModel, self).__call__(*args, **kwargs)
            if self.conditioning:
                # Reset alpha cache
                _, covar = output.representation()
                self.has_computed_alpha.fill_(0)
                self.alpha.resize_(gpytorch.posterior_strategy(covar).alpha_size())
                self.has_computed_lanczos.fill_(0)
                if gpytorch.functions.fast_pred_var:
                    lanczos_q_size, lanczos_t_size = gpytorch.posterior_strategy(covar).lanczos_size()
                    self.lanczos_q_mat.resize_(lanczos_q_size).zero_()
                    lanczos_t_mat_init = torch.eye(*lanczos_t_size).type_as(self.lanczos_t_mat)
                    self.lanczos_t_mat.resize_(lanczos_t_size).copy_(lanczos_t_mat_init)

        # Don't go through the output if we're training a variational inference model
        if self.training and not self.exact_inference:
            return output

        # Now go through the likelihood
        if isinstance(output, Variable) or isinstance(output, RandomVariable) or isinstance(output, LazyVariable):
            output = (output,)
        return self.likelihood(*output)
