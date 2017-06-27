import torch
import logging
from itertools import chain
from gpytorch.utils import pd_catcher, LBFGS
from torch.autograd import Variable
from .inference import Inference
from gpytorch.math.functions import AddDiag, Invmv
from gpytorch.math.modules import ExactGPMarginalLogLikelihood
from gpytorch.latent_distributions import GPDistribution
from gpytorch.likelihoods import GaussianLikelihood


class ExactGPInference(Inference):
    def __init__(self, observation_model):
        if not isinstance(observation_model.observation_model, GaussianLikelihood):
            raise RuntimeError('Exact GP inference is only defined for observation models of type GaussianLikelihoood')

        if not isinstance(observation_model.latent_distribution, GPDistribution):
            raise RuntimeError('Exact GP inference is only defined for latent distributions of type GPDistribution')

        super(ExactGPInference, self).__init__(observation_model)


    def run_(self, train_x, train_y, optimize=True, log_function=None, **optim_kwargs):
        
        likelihood = self.observation_model.observation_model
        latent_distribution = self.observation_model.latent_distribution

        # Optimize the latent distribution/likelihood hyperparameters
        # w.r.t. the marginal likelihood
        if optimize:
            marginal_log_likelihood = ExactGPMarginalLogLikelihood()
            parameters = list(chain(latent_distribution.parameters(), likelihood.parameters()))
            optimizer = LBFGS(parameters, line_search_fn='backtracking', **optim_kwargs)
            optimizer.n_iter = 0

            @pd_catcher(catch_function=lambda: Variable(torch.Tensor([10000])))
            def step_closure():
                optimizer.zero_grad()
                latent_distribution.zero_grad()
                likelihood.zero_grad()
                optimizer.n_iter += 1

                output = self.observation_model.forward(train_x)
                loss = -marginal_log_likelihood(output.covar(), train_y - output.mean())
                loss.backward()

                if log_function is not None:
                    logging.info(log_function(loss=loss, optimizer=optimizer, latent_distribution=latent_distribution, likelihood=likelihood))
                return loss

            optimizer.step(step_closure)

        train_covar = latent_distribution.forward_covar(train_x, train_x)
        train_covar = AddDiag()(train_covar, likelihood.log_noise.exp())

        
        # First, update the train_x buffer of latent_distribution
        latent_distribution.train_x.resize_as_(train_x.data).copy_(train_x.data)
        latent_distribution.train_x_var = Variable(latent_distribution.train_x)

        # Next, update train_covar_var with (K + \sigma^2 I)
        # K is training data kernel
        # sigma is likelihood noise
        latent_distribution.train_covar.resize_as_(train_covar.data).copy_(train_covar.data)
        latent_distribution.train_covar_var = Variable(latent_distribution.train_covar)
        
        # Finally, update alpha with (K + \sigma^2 I)^{-1} (y - \mu(x))
        alpha = Invmv()(train_covar, train_y - latent_distribution.forward_mean(train_x))
        latent_distribution.alpha.resize_as_(alpha.data).copy_(alpha.data)
        latent_distribution.alpha_var = Variable(latent_distribution.alpha)
        # Throw cholesky decomposition on the latent distribution, for efficiency
        latent_distribution.train_covar_var.chol_data = train_covar.chol_data

        return self.observation_model
