import math
import torch
import gpytorch
from torch.autograd import Variable
from gpytorch.parameters import MLEParameterGroup, BoundedParameter
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.inference import Inference
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function, but with KISS-GP let's use 100 training examples.
train_x = Variable(torch.linspace(0, 1, 100))
train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))

test_x = Variable(torch.linspace(0, 1, 51))
test_y = Variable(torch.sin(test_x.data * (2 * math.pi)))

# All tests that pass with the exact kernel should pass with the interpolated kernel.
class KissGPModel(gpytorch.ObservationModel):
    def __init__(self):
        super(KissGPModel,self).__init__(GaussianLikelihood())
        self.mean_module = ConstantMean()
        covar_module = RBFKernel()
        self.grid_covar_module = GridInterpolationKernel(covar_module, 50)
        self.params = MLEParameterGroup(
            log_noise=BoundedParameter(torch.Tensor([-2]),-5,5),
            log_lengthscale=BoundedParameter(torch.Tensor([0]),-3,5),
        )

    def forward(self,x):
        mean_x = self.mean_module(x, constant=Variable(torch.Tensor([0])))
        covar_x = self.grid_covar_module(x, log_lengthscale=self.params.log_lengthscale)

        latent_pred = GaussianRandomVariable(mean_x, covar_x)
        return latent_pred, self.params.log_noise

def test_kissgp_gp_mean_abs_error():
    prior_observation_model = KissGPModel()

    # Compute posterior distribution
    infer = Inference(prior_observation_model)
    posterior_observation_model = infer.run(train_x, train_y, max_inference_steps=1)

    test_preds = posterior_observation_model(test_x).mean()
    mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

    assert(mean_abs_error.data.squeeze()[0] < 0.1)