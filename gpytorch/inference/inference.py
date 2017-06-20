from copy import deepcopy

class Inference(object):
    def __init__(self, likelihood):
        self.likelihood = likelihood

    
    def run_(self, latent_distribution, train_x, train_y):
        raise NotImplementedError


    def run(self, latent_distribution, train_x, train_y):
        latent_distribution = deepcopy(latent_distribution)
        return self.run_(latent_distribution, train_x, train_y)
