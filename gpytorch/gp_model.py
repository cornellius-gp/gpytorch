import logging
import gpytorch
import torch
from torch import nn
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
        if not self.exact_inference:
            self.register_parameter('variational_mean', nn.Parameter(torch.Tensor()), bounds=(-1e4, 1e4))
            self.register_parameter('chol_variational_covar', nn.Parameter(torch.Tensor()), bounds=(-100, 100))

    def condition(self, train_inputs, train_target, **kwargs):
        """
        Conditions the model on data. After conditioning, the model functions
        in posterior mode rather than prior mode.

        The last input to args represents the target values (ys)
        All other inputs are input values (xs)

        NOTE:
        Calling this method also initializes variational parameters (if applicable)
        Calling this method also resets the alpha cache
        So if you are restoring a model from a saved state, call this method BEFORE you load
        the state dict

        Args: (Variables) inputs to condition on
        reset (bool) - reset variational parameters and alpha cache (default True)
        """
        if (isinstance(train_inputs, Variable) or isinstance(train_inputs, LazyVariable) or
                isinstance(train_inputs, RandomVariable)):
            train_inputs = train_inputs,

        res = super(GPModel, self).condition(train_inputs, train_target, **kwargs)

        # Initialize variational parameters (if applicable)
        if not self.exact_inference:
            if hasattr(self, 'inducing_points'):
                inducing_points = Variable(self.inducing_points)
            else:
                inducing_points = train_inputs[0]

            if len(train_inputs) > 1:
                raise RuntimeError('Variational inference currently only supports one input')
            inducing_output = super(GPModel, self).__call__(inducing_points)
            self.initialize_variational_parameters(inducing_output)

        return res

    @property
    def exact_inference(self):
        """
        Returns true if the model performs exact inference (vs. approximate inference)
        """
        return isinstance(self.likelihood, GaussianLikelihood)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def initialize_variational_parameters(self, output):
        """
        Initializes variational parameters
        """
        mean_init = output.mean().data
        chol_covar_init = torch.eye(len(mean_init)).type_as(mean_init)
        self.variational_mean.data.resize_as_(mean_init).copy_(mean_init)
        self.chol_variational_covar.data.resize_as_(chol_covar_init).copy_(chol_covar_init)
        return self

    def marginal_log_likelihood(self, output, target):
        """
        Returns the marginal log likelihood of the data

        Args:
        - output: (GaussianRandomVariable) - the output of the model
        - target: (Variable) - target
        """
        mean, covar = output.representation()

        # Exact inference
        if self.exact_inference:
            return gpytorch.exact_gp_marginal_log_likelihood(covar, target - mean)

        # Approximate inference
        else:
            # Get inducing points
            if not hasattr(self, 'train_inputs'):
                raise RuntimeError('Must condition on data.')

            train_x = self.train_inputs[0]
            if hasattr(self, 'inducing_points'):
                inducing_points = Variable(self.inducing_points)
            else:
                inducing_points = train_x

            chol_var_covar = self.chol_variational_covar.triu()
            # Negate each row with a negative diagonal (the Cholesky decomposition
            # of a matrix requires that the diagonal elements be positive).
            inside = chol_var_covar.diag().sign().unsqueeze(1).expand_as(chol_var_covar).triu()
            chol_var_covar = chol_var_covar.mul(inside)

            _, train_covar = output.representation()
            inducing_output = super(GPModel, self).__call__(inducing_points)
            inducing_mean, inducing_covar = inducing_output.representation()

            train_covar = gpytorch.add_jitter(train_covar)
            mcll_strategy = gpytorch.posterior_strategy(train_covar)
            log_likelihood = mcll_strategy.monte_carlo_log_likelihood(self.likelihood.log_probability,
                                                                      target,
                                                                      self.variational_mean,
                                                                      chol_var_covar)

            inducing_covar = gpytorch.add_jitter(inducing_covar)
            kl_divergence = gpytorch.mvn_kl_divergence(self.variational_mean,
                                                       chol_var_covar, inducing_mean, inducing_covar)

            res = log_likelihood.squeeze() - kl_divergence
            return res

    @property
    def needs_grid(self):
        return True

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

            n_train = len(train_xs[0])
            full_inputs = [torch.cat([train_x, input]) for train_x, input in zip(train_xs, args)]
            full_output = super(GPModel, self).__call__(*full_inputs, **kwargs)
            full_mean, full_covar = full_output.representation()

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
                # Ensure variational parameters have been initalized
                if not self.variational_mean.numel():
                    raise RuntimeError('Variational parameters have not been initalized.'
                                       'Condition on data.')

                # Get inducing points
                if hasattr(self, 'inducing_points'):
                    inducing_points = Variable(self.inducing_points)
                else:
                    inducing_points = train_xs[0]

                n_induc = len(inducing_points)
                full_input = torch.cat([inducing_points, args[0]])
                full_output = super(GPModel, self).__call__(full_input, **kwargs)
                full_mean, full_covar = full_output.representation()

                test_mean = full_mean[n_induc:]
                induc_induc_covar = full_covar[:n_induc, :n_induc]
                induc_test_covar = full_covar[:n_induc, n_induc:]
                test_induc_covar = full_covar[n_induc:, :n_induc]
                test_test_covar = full_covar[n_induc:, n_induc:]

                # Calculate posterior components
                if not self.has_computed_alpha[0]:
                    alpha_strategy = gpytorch.posterior_strategy(induc_induc_covar)
                    alpha = alpha_strategy.variational_posterior_alpha(self.variational_mean)
                    self.alpha.copy_(alpha.data)
                    self.has_computed_alpha.fill_(1)
                else:
                    alpha = Variable(self.alpha)
                mean_strategy = gpytorch.posterior_strategy(test_induc_covar)
                test_mean = mean_strategy.variational_posterior_mean(alpha)
                covar_strategy = gpytorch.posterior_strategy(test_induc_covar)
                test_covar = covar_strategy.variational_posterior_covar(induc_test_covar, self.chol_variational_covar,
                                                                        test_test_covar, induc_induc_covar)
                output = GaussianRandomVariable(test_mean, test_covar)

        # Training or Prior mode
        else:
            output = super(GPModel, self).__call__(*args, **kwargs)
            # Add some jitter
            if not self.exact_inference:
                mean, covar = output.representation()
                covar = gpytorch.add_jitter(covar)
                output = GaussianRandomVariable(mean, covar)

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

        # Now go through the likelihood
        if isinstance(output, Variable) or isinstance(output, RandomVariable) or isinstance(output, LazyVariable):
            output = (output,)
        return self.likelihood(*output)
