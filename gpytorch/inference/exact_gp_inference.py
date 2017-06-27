import torch
import logging
from itertools import chain
from gpytorch.utils import pd_catcher, LBFGS
from torch.autograd import Variable
from .inference import Inference
from gpytorch.math.functions import AddDiag, Invmv
from gpytorch.math.modules import ExactGPMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import Kernel, PosteriorKernel
from gpytorch.means import Mean, PosteriorMean


class ExactGPInference(Inference):
    def __init__(self, observation_model):
        if not isinstance(observation_model.observation_model, GaussianLikelihood):
            raise RuntimeError('Exact GP inference is only defined for observation models of type GaussianLikelihoood')

        if len(observation_model.latent_distributions) != 2 or \
            set([dist.__class__ for dist in observation_model.latent_distributions]) == {Mean, Kernel}:
            raise RuntimeError('Observation model must have exactly 2 latent distributions: a Mean and a Kernel')

        super(ExactGPInference, self).__init__(observation_model)


    def run_(self, train_x, train_y, optimize=True, log_function=None, **optim_kwargs):
        likelihood = self.observation_model.observation_model

        mean_module_name, mean_module = [(name, module) for (name, module) in self.observation_model.latent_distributions.items() if isinstance(module,Mean)][0]
        kernel_module_name, kernel_module = [(name, module) for (name, module) in self.observation_model.latent_distributions.items() if isinstance(module,Kernel)][0]

        # Optimize the latent distribution/likelihood hyperparameters
        # w.r.t. the marginal likelihood
        if optimize:
            marginal_log_likelihood = ExactGPMarginalLogLikelihood()
            parameters = self.observation_model.parameters()
            optimizer = LBFGS(parameters, line_search_fn='backtracking', **optim_kwargs)
            optimizer.n_iter = 0

            @pd_catcher(catch_function=lambda: Variable(torch.Tensor([10000])))
            def step_closure():
                optimizer.zero_grad()
                self.observation_model.zero_grad()
                optimizer.n_iter += 1

                output = self.observation_model.forward(train_x)

                loss = -marginal_log_likelihood(output.covar(), train_y - output.mean())
                loss.backward()

                if log_function is not None:
                    logging.info(log_function(loss=loss, optimizer=optimizer, observation_model=self.observation_model))
                return loss

            optimizer.step(step_closure)

        posterior_kernel_module = PosteriorKernel(kernel_module,train_x,likelihood.log_noise)
        posterior_mean_module = PosteriorMean(mean_module,kernel_module,train_x,train_y,likelihood.log_noise)
        setattr(self.observation_model, mean_module_name, posterior_mean_module)
        setattr(self.observation_model, kernel_module_name, posterior_kernel_module)

        return self.observation_model
