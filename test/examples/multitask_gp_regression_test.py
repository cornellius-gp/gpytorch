import math
import torch
import gpytorch
from torch import optim
from torch.autograd import Variable
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


class MultitaskGPModel(gpytorch.GPModel):
    def __init__(self):
        likelihood = GaussianLikelihood(log_noise_bounds=(-6, 6))
        super(MultitaskGPModel, self).__init__(likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-6, 6))
        self.task_covar_module = IndexKernel(n_tasks=2, rank=1, covar_factor_bounds=(-6, 6), log_var_bounds=(-6, 6))

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)
        covar_xi = covar_x.mul(covar_i)
        return GaussianRandomVariable(mean_x, covar_xi)


def test_multitask_gp_mean_abs_error():
    prior_gp_model = MultitaskGPModel()

    # Compute posterior distribution
    infer = Inference(prior_gp_model)
    posterior_gp_model = infer.run((torch.cat([train_x, train_x]), torch.cat([y1_inds, y2_inds])),
                                   torch.cat([train_y1, train_y2]))

    # Optimize the model
    posterior_gp_model.train()
    optimizer = optim.Adam(posterior_gp_model.parameters(), lr=0.1)
    optimizer.n_iter = 0
    for i in range(100):
        optimizer.zero_grad()
        output = posterior_gp_model(torch.cat([train_x, train_x]), torch.cat([y1_inds, y2_inds]))
        loss = -posterior_gp_model.marginal_log_likelihood(output, torch.cat([train_y1, train_y2]))
        loss.backward()
        optimizer.n_iter += 1
        optimizer.step()

    # Test the model
    posterior_gp_model.eval()
    test_preds_task_1 = posterior_gp_model(test_x, y1_inds_test).mean()
    mean_abs_error_task_1 = torch.mean(torch.abs(test_y1 - test_preds_task_1))

    assert(mean_abs_error_task_1.data.squeeze()[0] < 0.05)

    test_preds_task_2 = posterior_gp_model(test_x, y2_inds_test).mean()
    mean_abs_error_task_2 = torch.mean(torch.abs(test_y2 - test_preds_task_2))

    assert(mean_abs_error_task_2.data.squeeze()[0] < 0.05)
