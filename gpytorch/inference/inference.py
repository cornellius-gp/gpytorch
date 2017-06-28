from ..likelihoods import Likelihood, GaussianLikelihood
from ..random_variables import GaussianRandomVariable
from .exact_gp_inference import ExactGPInference
from gpytorch import ObservationModel
from copy import deepcopy


class Inference(object):
    def __init__(self, observation_model):
    	self.observation_model = observation_model
        self.inference_engine = None

        if isinstance(self.observation_model.observation_model, ObservationModel):
            self.observation_model_inference = Inference(self.observation_model.observation_model)
        else:
            self.observation_model_inference = None

    
    def run_(self, train_x, train_y, **kwargs):
        repeat = True
        while repeat:
            output = self.forward(train_x, train_y, **kwargs)
            new_observation_model, repeat = self.step(output)
            self.observation_model = new_observation_model
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
