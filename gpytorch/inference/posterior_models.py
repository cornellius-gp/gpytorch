import torch
import logging
from itertools import chain
from gpytorch.utils import pd_catcher
from torch.autograd import Variable
from torch.nn import Parameter
from gpytorch import ObservationModel
from gpytorch.parameters import MLEParameterGroup
from gpytorch.random_variables import GaussianRandomVariable


class _GPPosterior(ObservationModel):
    def update_data(self, train_xs, train_y):
        """
        Updates this model's training data internally to use the supplied
        train_xs and train_y.

        This method is not intended to update the parameters of the model, that
        is the job of Inference. Only the buffers storing data should be
        updated here.
        """
        pass

    def forward(self, *inputs, **params):
        """
        Given a set of inputs x, returns the predictive posterior distribution
        for the latent function, p(f*|D, x*), where D is the training data as
        set using update_data.
        """
        pass

    def marginal_log_likelihood(self, output, train_y):
        """
        Returns the log marginal likelihood of the data (for exact inference)
        or some lower bound on it (for variational inference).  Inference uses
        this function to optimize or perform sampling for all parameters of the
        model.
        """
        pass


class _VariationalGPPosterior(_GPPosterior):
    def __init__(self, gp_observation_model, inducing_points, train_xs=None, train_y=None):
        super(_VariationalGPPosterior, self).__init__(gp_observation_model.observation_model)
        self.gp_observation_model = gp_observation_model
        self.inducing_points = inducing_points

        if train_xs is not None and train_y is not None:
            self.update_data(train_xs, train_y)

        num_inducing = len(self.inducing_points[0])
        self.variational_parameters = MLEParameterGroup(
            variational_mean=Parameter(torch.randn(num_inducing)),
            chol_variational_covar=Parameter(torch.randn(num_inducing, num_inducing).triu_()),
        )

    def update_data(self, train_xs, train_y):
        if isinstance(train_xs, Variable) or isinstance(train_xs, torch._TensorBase):
            train_xs = (train_xs, )
        train_xs = [input.data if isinstance(input, Variable) else input for input in train_xs]
        train_y = train_y.data if isinstance(train_y, Variable) else train_y

        self.train_xs = []
        for i, train_x in enumerate(train_xs):
            if hasattr(self, 'train_x_%d' % i):
                getattr(self, 'train_x_%d').resize_as_(train_x).copy_(train_x)
            else:
                self.register_buffer('train_x_%d' % i, train_x)
            self.train_xs.append(getattr(self, 'train_x_%d' % i))

        if hasattr(self, 'train_y'):
            self.train_y.resize_as_(train_y).copy_(train_y)
        else:
            self.register_buffer('train_y', train_y)
        return self

    def forward(self, *inputs, **params):
        has_posterior = len(self.train_xs[0]) if hasattr(self, 'train_xs') else 0

        n = len(self.inducing_points[0])

        if has_posterior:
            inducing_point_vars = [Variable(train_x) for train_x in self.train_xs]
            full_inputs = [torch.cat([inducing_point_var, input])
                           for inducing_point_var, input in zip(inducing_point_vars, inputs)]
        else:
            full_inputs = inputs

        gaussian_rv_output = self.gp_observation_model.forward(*full_inputs, **params)
        full_mean, full_covar = gaussian_rv_output.representation()

        if not has_posterior:
            test_mean = full_mean
            test_covar = full_covar
        else:
            train_train_covar = full_covar[:n, :n]
            test_train_covar = full_covar[n:, :n]
            train_test_covar = full_covar[:n, n:]

            alpha = gpytorch.invmv(train_train_covar, self.variational_parameters.variational_mean)
            test_mean = torch.mv(test_train_covar, alpha)

            chol_covar = self.variational_parameters.chol_variational_covar
            variational_covar = chol_covar.t().mm(chol_covar)

            test_covar = variational_covar - train_train_covar

            # test_covar = K_{mn}K_{nn}^{-1}(S - K_{nn})
            test_covar = torch.mm(test_train_covar, gpytorch.invmm(train_train_covar, test_covar))

            # right_factor = K_{nn}^{-1}K_{nm}
            right_factor = gpytorch.invmm(train_train_covar, train_test_covar)

            # test_covar = K_{mn}K_{nn}^{-1}(S - K_{nn})K_{nn}^{-1}K_{nm}
            test_covar = full_covar[n:, n:] + test_covar.mm(right_factor)

        return GaussianRandomVariable(test_mean, test_covar)

    def marginal_log_likelihood(self, output, train_y, num_samples=5):
        chol_var_covar = self.variational_parameters.chol_variational_covar.triu()

        # Negate each row with a negative diagonal (the Cholesky decomposition
        # of a matrix requires that the diagonal elements be positive).
        chol_var_covar = chol_var_covar.mul(chol_var_covar.diag().sign().unsqueeze(1).expand_as(chol_var_covar).triu())

        inducing_mean, inducing_covar = output.representation()
        num_inducing = len(inducing_mean)

        epsilon = Variable(torch.randn(num_inducing, num_samples))
        samples = chol_var_covar.mm(epsilon)
        samples = samples + self.variational_parameters.variational_mean.unsqueeze(1).expand_as(samples)
        log_likelihood = self.observation_model.log_probability(samples, train_y)

        kl_divergence = gpytorch.mvn_kl_divergence(self.variational_parameters.variational_mean,
                                                   chol_var_covar, inducing_mean, inducing_covar)

        return log_likelihood.squeeze() - kl_divergence


class _ExactGPPosterior(_GPPosterior):
    def __init__(self, gp_observation_model, train_xs=None, train_y=None):
        super(_ExactGPPosterior, self).__init__(gp_observation_model.observation_model)
        self.gp_observation_model = gp_observation_model

        # Buffers for conditioning on data
        if train_xs is not None and train_y is not None:
            self.update_data(train_xs, train_y)

    def update_data(self, train_xs, train_y):
        if isinstance(train_xs, Variable) or isinstance(train_xs, torch._TensorBase):
            train_xs = (train_xs, )
        train_xs = [input.data if isinstance(input, Variable) else input for input in train_xs]
        train_y = train_y.data if isinstance(train_y, Variable) else train_y

        self.train_xs = []
        for i, train_x in enumerate(train_xs):
            if hasattr(self, 'train_x_%d' % i):
                getattr(self, 'train_x_%d').resize_as_(train_x).copy_(train_x)
            else:
                self.register_buffer('train_x_%d' % i, train_x)
            self.train_xs.append(getattr(self, 'train_x_%d' % i))

        if hasattr(self, 'train_y'):
            self.train_y.resize_as_(train_y).copy_(train_y)
        else:
            self.register_buffer('train_y', train_y)
        return self

    def forward(self, *inputs, **params):
        n = len(self.train_xs[0]) if hasattr(self, 'train_xs') else 0

        # Compute mean and full data (train/test) covar
        if n:
            train_x_vars = [Variable(train_x) for train_x in self.train_xs]
            full_inputs = [torch.cat([train_x_var, input]) for train_x_var, input in zip(train_x_vars, inputs)]
        else:
            full_inputs = inputs
        gaussian_rv_output, log_noise = self.gp_observation_model.forward(*full_inputs, **params)
        full_mean, full_covar = gaussian_rv_output.representation()

        # Get mean/covar components
        test_mean = full_mean[n:]
        test_test_covar = full_covar[n:, n:]

        # If there's data, use it
        if n:
            train_y_var = Variable(self.train_y)
            train_mean = full_mean[:n]
            train_train_covar = gpytorch.add_diag(full_covar[:n, :n], log_noise.exp())
            test_train_covar = full_covar[n:, :n]
            train_test_covar = full_covar[:n, n:]

            # Update test mean
            alpha = gpytorch.invmv(train_train_covar, train_y_var - train_mean)
            test_mean = test_mean.add(torch.mv(test_train_covar, alpha))

            # Update test-test covar
            test_test_covar_correction = torch.mm(test_train_covar, gpytorch.invmm(train_train_covar, train_test_covar))
            test_test_covar = test_test_covar.sub(test_test_covar_correction)

        return GaussianRandomVariable(test_mean, test_test_covar), log_noise

    def marginal_log_likelihood(self, output, train_y):
        mean, covar = output.representation()
        return gpytorch.exact_gp_marginal_log_likelihood(covar, train_y - mean)
