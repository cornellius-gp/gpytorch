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
        self.inducing_points = None

        self.register_buffer('alpha', torch.Tensor())
        if not self.exact_inference:
            self.register_parameter('variational_mean', nn.Parameter(torch.Tensor()), bounds=(-1e4, 1e4))
            self.register_parameter('chol_variational_covar', nn.Parameter(torch.Tensor()), bounds=(-100, 100))

    def condition(self, *args, **kwargs):
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

        if len(args) < 2:
            raise RuntimeError('A GPModel must condition on at least two data (inputs and targets)!')

        # Check to be sure that this method was called before loading state dict
        if not hasattr(self, 'train_data') and self.alpha.numel():
            logging.warning('You are conditioning on data after loading the state dict.'
                            'This has reset caches and variational parameters (if applicable).')

        reset = kwargs.get('reset', True)
        if reset:
            # Reset alpha cache
            self.alpha.resize_(0)

            # Reset variational parameters (if applicable)
            if not self.exact_inference:
                if len(args) != 2:
                    raise RuntimeError('Variational inference currently only works with one input.')
                self.variational_mean.data.resize_(0)
                self.chol_variational_covar.data.resize_(0)

        super(GPModel, self).condition(*args, **kwargs)

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
        chol_covar_init = torch.eye(len(mean_init))
        self.variational_mean.data.resize_as_(mean_init).copy_(mean_init)
        self.chol_variational_covar.data.resize_as_(chol_covar_init).copy_(chol_covar_init)
        return self

    def initialize_interpolation_grid(self, grid_size, grid_bounds):
        super(GPModel, self).initialize_interpolation_grid(grid_size, grid_bounds)
        self.grid_size = grid_size
        self.grid_bounds = grid_bounds
        grid = torch.zeros(len(grid_bounds), grid_size)
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[i] = torch.linspace(grid_bounds[i][0] - grid_diff,
                                     grid_bounds[i][1] + grid_diff,
                                     grid_size)
        self.inducing_points = torch.zeros(int(pow(grid_size, len(grid_bounds))), len(grid_bounds))
        for i in range(self.inducing_points.size()[0]):
            for j in range(len(grid_bounds)):
                self.inducing_points[i][j] = grid[j][int(i / pow(grid_size, j)) % pow(grid_size, j + 1)]
        self.inducing_points = Variable(self.inducing_points)
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
            if not hasattr(self, 'train_data'):
                raise RuntimeError('Must condition on data.')

            train_x = self.train_data[0]
            inducing_points = self.inducing_points
            if inducing_points is None:
                inducing_points = train_x

            # Initialize variational parameters - if necessary
            if not self.variational_mean.numel():
                inducing_output = super(GPModel, self).__call__(inducing_points)
                self.initialize_variational_parameters(inducing_output)

            chol_var_covar = self.chol_variational_covar.triu()
            # Negate each row with a negative diagonal (the Cholesky decomposition
            # of a matrix requires that the diagonal elements be positive).
            inside = chol_var_covar.diag().sign().unsqueeze(1).expand_as(chol_var_covar).triu()
            chol_var_covar = chol_var_covar.mul(inside)

            _, train_covar = output.representation()
            inducing_output = super(GPModel, self).__call__(inducing_points)
            inducing_mean, inducing_covar = inducing_output.representation()

            train_covar = gpytorch.add_jitter(train_covar)
            log_likelihood = gpytorch.monte_carlo_log_likelihood(self.likelihood.log_probability,
                                                                 target,
                                                                 self.variational_mean,
                                                                 chol_var_covar,
                                                                 train_covar)

            inducing_covar = gpytorch.add_jitter(inducing_covar)
            kl_divergence = gpytorch.mvn_kl_divergence(self.variational_mean,
                                                       chol_var_covar, inducing_mean, inducing_covar)

            res = log_likelihood.squeeze() - kl_divergence
            return res

    def train(self, mode=True):
        if mode:
            self.alpha.resize_(0)
        return super(GPModel, self).train(mode)

    def __call__(self, *args, **kwargs):
        output = None

        # Posterior mode
        if self.posterior:
            train_xs = self.train_data[:-1]
            train_y = self.train_data[-1]
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
                if not self.alpha.numel():
                    alpha_strategy = gpytorch.posterior_strategy(train_train_covar)
                    alpha = alpha_strategy.exact_posterior_alpha(train_mean, train_y)
                    self.alpha.resize_as_(alpha.data).copy_(alpha.data)
                else:
                    alpha = Variable(self.alpha)
                mean_strategy = gpytorch.posterior_strategy(test_train_covar)
                test_mean = mean_strategy.exact_posterior_mean(test_mean, alpha)
                covar_strategy = gpytorch.posterior_strategy(train_train_covar)
                test_covar = covar_strategy.exact_posterior_covar(test_train_covar, train_test_covar, test_test_covar)
                output = GaussianRandomVariable(test_mean, test_covar)

            # Approximate inference
            else:
                # Ensure variational parameters have been initalized
                if not self.variational_mean.numel():
                    raise RuntimeError('Variational parameters have not been initalized.'
                                       'Train the model or load a state dict.')

                # Get inducing points
                inducing_points = self.inducing_points
                if inducing_points is None:
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
                if not self.alpha.numel():
                    alpha_strategy = gpytorch.posterior_strategy(induc_induc_covar)
                    alpha = alpha_strategy.variational_posterior_alpha(self.variational_mean)
                    self.alpha.resize_as_(alpha.data).copy_(alpha.data)
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

        # Now go through the likelihood
        if isinstance(output, Variable) or isinstance(output, RandomVariable) or isinstance(output, LazyVariable):
            output = (output,)
        return self.likelihood(*output)
