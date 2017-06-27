from copy import deepcopy

class Inference(object):
    def __init__(self, observation_model):
    	self.observation_model = observation_model

    
    def run_(self, train_x, train_y, **kwargs):
        raise NotImplementedError


    def run(self, train_x, train_y, **kwargs):
    	orig_observation_model = self.observation_model
        self.observation_model = deepcopy(self.observation_model)
        new_observation_model  = self.run_(train_x, train_y, **kwargs)
        self.observation_model = orig_observation_model
        return new_observation_model
