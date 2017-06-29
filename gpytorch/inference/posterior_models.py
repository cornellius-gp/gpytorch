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
    def __init__(self, gp_observation_model, train_x=None, train_y=None):
        super(_ExactGPPosterior, self).__init__(gp_observation_model.observation_model)
        self.gp_observation_model = gp_observation_model

        # Buffers for conditioning on data
        train_x = train_x or torch.Tensor()
        train_y = train_y or torch.Tensor()
        if isinstance(train_x, Variable):
            train_x = train_x.data
        if isinstance(train_y, Variable):
            train_y = train_y.data
        self.register_buffer('train_x', train_x)
        self.register_buffer('train_y', train_y)


    def forward(self, input, **params):
        n = len(self.train_x)
        m = len(input)

        # Wrap train x and train y in variables
        train_x_var = Variable(self.train_x)
        train_y_var = Variable(self.train_y)

        # Compute mean and full data (train/test) covar
        if n:
            full_input = torch.cat([train_x_var, input])
        else:
            full_input = input
        gaussian_rv_output, log_noise = self.gp_observation_model.forward(full_input, **params)
        full_mean, full_covar = gaussian_rv_output.representation()

        # Get mean/covar components
        test_mean = full_mean[n:]
        test_test_covar = full_covar[n:, n:]

        # If there's data, use it
        if n:
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
