import torch
from .distribution import Distribution

class ObservationModel(Distribution):
	def __init__(self,observation_model,latent_distribution):
		super(ObservationModel,self).__init__()
		self.observation_model = observation_model
		self.latent_distribution = latent_distribution
