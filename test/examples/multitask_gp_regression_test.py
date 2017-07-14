import math
import torch
import gpytorch

from torch.autograd import Variable
from torch.nn import Parameter
from gpytorch.parameters import MLEParameterGroup
from gpytorch.kernels import RBFKernel, IndexKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.inference import Inference
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function
train_x = Variable(torch.linspace(0, 1, 11))
y1_inds = Variable(torch.zeros(11).long())
y2_inds = Variable(torch.ones(11).long())
train_y1 = Variable(torch.sin(train_x.data * (2 * math.pi)))
train_y2 = Variable(torch.cos(train_x.data * (2 * math.pi)))

test_x = Variable(torch.linspace(0, 1, 51))
y1_inds_test = Variable(torch.zeros(51).long())
y2_inds_test = Variable(torch.ones(51).long())
test_y1 = Variable(torch.sin(test_x.data * (2 * math.pi)))
test_y2 = Variable(torch.cos(test_x.data * (2 * math.pi)))


class MultitaskGPModel(gpytorch.ObservationModel):
    def __init__(self):
        super(MultitaskGPModel, self).__init__(GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel()
        self.task_covar_module = IndexKernel()
        self.model_params = MLEParameterGroup(
            constant_mean=Parameter(torch.randn(1)),
            log_noise=Parameter(torch.randn(1)),
            log_lengthscale=Parameter(torch.randn(1)),
        )
        self.task_params = MLEParameterGroup(
            task_matrix=Parameter(torch.randn(2, 1)),
            task_log_vars=Parameter(torch.randn(2)),
        )

    def forward(self, x, i):
        mean_x = self.mean_module(x, constant=self.model_params.constant_mean)

        covar_x = self.covar_module(x, log_lengthscale=self.model_params.log_lengthscale)
        covar_i = self.task_covar_module(i,
                                         index_covar_factor=self.task_params.task_matrix,
                                         index_log_var=self.task_params.task_log_vars)

        covar_xi = covar_x.mul(covar_i)

        latent_pred = GaussianRandomVariable(mean_x, covar_xi)
        return latent_pred, self.model_params.log_noise


def test_multitask_gp_mean_abs_error():
    prior_observation_model = MultitaskGPModel()

    # Compute posterior distribution
    infer = Inference(prior_observation_model)
    posterior_observation_model = infer.run((torch.cat([train_x, train_x]),
                                            torch.cat([y1_inds, y2_inds])),
                                            torch.cat([train_y1, train_y2]))

    test_preds_task_1 = posterior_observation_model(test_x, y1_inds_test).mean()
    mean_abs_error_task_1 = torch.mean(torch.abs(test_y1 - test_preds_task_1))

    assert(mean_abs_error_task_1.data.squeeze()[0] < 0.01)

    test_preds_task_2 = posterior_observation_model(test_x, y2_inds_test).mean()
    mean_abs_error_task_2 = torch.mean(torch.abs(test_y2 - test_preds_task_2))

    assert(mean_abs_error_task_2.data.squeeze()[0] < 0.01)
