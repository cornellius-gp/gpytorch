from ..likelihoods import Likelihood, GaussianLikelihood
from ..random_variables import GaussianRandomVariable
from .posterior_models import _ExactGPPosterior
from gpytorch import ObservationModel
from copy import deepcopy
from torch.autograd import Variable


class Inference(object):
    def __init__(self, observation_model):
    	self.observation_model = observation_model
        self.inference_engine = None

        if isinstance(self.observation_model.observation_model, ObservationModel):
            self.observation_model_inference = Inference(self.observation_model.observation_model)
        else:
            self.observation_model_inference = None


    def run_(self, train_x, train_y, **kwargs):
        if isinstance(self.observation_model, Likelihood):
            raise RuntimeError('Likelihood should not have an inference engine')

        # Replace observation models with posterior versions
        likelihood = self.observation_model.observation_model
        if isinstance(likelihood, GaussianLikelihood):
            output = self.observation_model.forward(train_x, **kwargs)
            if len(output) == 2 and isinstance(output[0], GaussianRandomVariable):
                if not isinstance(self.observation_model,_ExactGPPosterior):
                    self.observation_model = _ExactGPPosterior(self.observation_model)
                else:
                    raise RuntimeError('Updating existing GP posteriors is not yet supported.')
            else:
                raise RuntimeError('Unknown inference type for observation model:\n%s' % repr(observation_model))
        else:
            raise RuntimeError('Inference for likelihoods other than Gaussian not yet supported.')

        # Define a function to get the marginal log likelihood of the model, given data/parameter values
        def log_likelihood_closure():
            self.observation_model.zero_grad()
            output = self.observation_model(train_x)
            return likelihood.marginal_log_likelihood(output, train_y)

        # Update all parameter groups
        param_groups = list(self.observation_model.parameter_groups())
        if len(param_groups) > 1:
            raise RuntimeError('Inference for multiple parameter groups not yet supported.')

        for param_group in param_groups:
            param_group.update(log_likelihood_closure)

        # Add the data
        self.observation_model.train_x.resize_as_(train_x.data).copy_(train_x.data)
        self.observation_model.train_y.resize_as_(train_y.data).copy_(train_y.data)

        return self.observation_model


    def run(self, train_x, train_y, **kwargs):
        orig_observation_model = self.observation_model
        self.observation_model = deepcopy(self.observation_model)
        new_observation_model  = self.run_(train_x, train_y, **kwargs)
        self.observation_model = orig_observation_model
        return new_observation_model


    def forward(self, train_x, train_y, **params):
        if self.inference_engine is None:
            if isinstance(self.observation_model, Likelihood):
                raise RuntimeError('Likelihood should not have an inference engine')

            elif isinstance(self.observation_model.observation_model, GaussianLikelihood):
                output = self.observation_model.forward(train_x, **params)
                if len(output) == 2 and isinstance(output[0], GaussianRandomVariable):
                    self.inference_engine = ExactGPInference(self.observation_model)
                else:
                    raise RuntimeError('Unknown inference type for observation model:\n%s' % repr(observation_model))

        return self.inference_engine.forward(train_x, train_y, **params)


    def step(self, output):
        return self.inference_engine.step(output)
