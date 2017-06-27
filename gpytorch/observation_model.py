import torch
from .distribution import Distribution

class ObservationModel(Distribution):
	def __init__(self,observation_model,latent_distributions):
		super(ObservationModel,self).__init__()
		self.observation_model = observation_model
		self.latent_distributions = latent_distributions
		for name, latent_distribution in latent_distributions.items():
			self.add_module(name, latent_distribution)


	def forward(self, *args, **kwargs):
		raise NotImplementedError