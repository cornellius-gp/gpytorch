import torch
import logging
from itertools import chain
from gpytorch.utils import pd_catcher, LBFGS
from torch.autograd import Variable
from gpytorch.math.functions import AddDiag, Invmv, Invmm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from gpytorch import ObservationModel
from gpytorch.random_variables import GaussianRandomVariable


class _ExactGPPosterior(ObservationModel):
    def __init__(self, gp_observation_model, train_xs=None, train_y=None):
        super(_ExactGPPosterior, self).__init__(gp_observation_model.observation_model)
        self.gp_observation_model = gp_observation_model

        # Buffers for conditioning on data
        if train_xs is not None and train_y is not None:
            self.update_data(train_xs, train_y)

    def update_data(self, train_xs, train_y):
        if isinstance(train_xs, Variable) or isinstance(train_xs, torch._TensorBase):
            train_xs = (train_xs,)
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
        m = len(inputs[0])

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
            train_train_covar = AddDiag()(full_covar[:n, :n], log_noise.exp())
            test_train_covar = full_covar[n:, :n]
            train_test_covar = full_covar[:n, n:]

            # Update test mean
            alpha = Invmv()(train_train_covar, train_y_var - train_mean)
            test_mean = test_mean.add(torch.mv(test_train_covar, alpha))

            # Update test-test covar
            test_test_covar_correction = torch.mm(test_train_covar, Invmm()(train_train_covar, train_test_covar))
            test_test_covar = test_test_covar.sub(test_test_covar_correction)

        return GaussianRandomVariable(test_mean, test_test_covar), log_noise
