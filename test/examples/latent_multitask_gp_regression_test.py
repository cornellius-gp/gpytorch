import math
import torch
import gpytorch

from torch.autograd import Variable
from gpytorch.kernels import RBFKernel, IndexKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.inference import Inference
from gpytorch import ObservationModel
from gpytorch.random_variables import GaussianRandomVariable, BatchRandomVariables, CategoricalRandomVariable
from gpytorch.parameters import MLEParameterGroup, CategoricalMCParameterGroup
from torch.nn import Parameter

# Training data with 3 visible tasks, but the model should learn that task 11 and 12 should be grouped together.
train_x = Variable(torch.linspace(0, 1, 11))
y11_inds = Variable(torch.zeros(11).long())
y12_inds = Variable(torch.ones(11).long())
y2_inds = Variable(2*torch.ones(11).long())
train_y11 = Variable(torch.sin(train_x.data * (2 * math.pi)) +  torch.randn(train_x.size()) * 0.01)
train_y12 = Variable(torch.sin(train_x.data * (2 * math.pi)) +  torch.randn(train_x.size()) * 0.01)
train_y2 = Variable(torch.cos(train_x.data * (2 * math.pi))  +  torch.randn(train_x.size()) * 0.01)

test_x = Variable(torch.linspace(0, 1, 51))
y11_inds_test = Variable(torch.zeros(51).long())
y12_inds_test = Variable(torch.ones(51).long())
y2_inds_test = Variable(2*torch.ones(51).long())
test_y11 = Variable(torch.sin(test_x.data * (2 * math.pi)))
test_y12 = Variable(torch.sin(test_x.data * (2 * math.pi)))
test_y2 = Variable(torch.cos(test_x.data * (2 * math.pi)))

class LatentMultitaskGPModel(gpytorch.ObservationModel):
    def __init__(self,num_task_samples):
        super(LatentMultitaskGPModel,self).__init__(GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel()
        self.task_covar_module = IndexKernel()

        self.model_params = MLEParameterGroup(
            constant_mean=Parameter(torch.randn(1)),
            log_lengthscale=Parameter(torch.randn(1)),
            log_noise=Parameter(torch.randn(1)),
        )

        self.task_params = MLEParameterGroup(
            task_matrix=Parameter(torch.randn(2,1)),
            task_log_vars=Parameter(torch.randn(2)),
        )

        task_prior = CategoricalRandomVariable(0.5*torch.ones(2))
        self.latent_tasks = CategoricalMCParameterGroup(
            task_assignments=BatchRandomVariables(task_prior,3),
        )

        self.num_task_samples = num_task_samples
        self.latent_tasks.set_options(num_samples=num_task_samples)

    def forward(self,x,i):
        n = len(x)
        mean_x = self.mean_module(x, constant=self.model_params.constant_mean)
        covar_x = self.covar_module(x, log_lengthscale=self.model_params.log_lengthscale)
        covar_i = Variable(torch.zeros(n,n))

        for j in range(self.num_task_samples):
            task_assignments = self.latent_tasks.task_assignments.sample()
            task_assignments = task_assignments.index_select(0,i)
            covar_ji = self.task_covar_module(task_assignments,
                                             index_covar_factor=self.task_params.task_matrix,
                                             index_log_var=self.task_params.task_log_vars)
            covar_i += covar_ji.mul_(1./self.num_task_samples)

        covar_xi = covar_x.mul(covar_i)
        latent_pred = GaussianRandomVariable(mean_x, covar_xi)
        return latent_pred, self.model_params.log_noise

def test_latent_multitask_gp_mean_abs_error():
    prior_observation_model = LatentMultitaskGPModel(num_task_samples=3)

    # Compute posterior distribution
    infer = Inference(prior_observation_model)
    posterior_observation_model = infer.run((torch.cat([train_x,train_x,train_x]),
                                             torch.cat([y11_inds,y12_inds,y2_inds])),
                                            torch.cat([train_y11,train_y12,train_y2]),
                                            max_inference_steps=5
                                       )
    observed_pred_y11 = posterior_observation_model(test_x,y11_inds_test)
    mean_abs_error_task_11 = torch.mean(torch.abs(test_y11 - observed_pred_y11.mean()))
    assert(mean_abs_error_task_11.data.squeeze()[0] < 0.05)

    observed_pred_y12 = posterior_observation_model(test_x,y12_inds_test)
    mean_abs_error_task_12 = torch.mean(torch.abs(test_y12 - observed_pred_y12.mean()))
    assert(mean_abs_error_task_12.data.squeeze()[0] < 0.05)

    observed_pred_y2 = posterior_observation_model(test_x,y2_inds_test)
    mean_abs_error_task_2 = torch.mean(torch.abs(test_y2 - observed_pred_y2.mean()))
    assert(mean_abs_error_task_2.data.squeeze()[0] < 0.01)
