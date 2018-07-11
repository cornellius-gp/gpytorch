from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .acquisition_function import AcquisitionFunction
from torch.distributions.normal import Normal


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, gp_model, best_y):
        super(ExpectedImprovement, self).__init__(gp_model)
        self.best_y = best_y

    def forward(self, candidate_set):
        self.grid_size = 10000

        self.gp_model.eval()
        self.gp_model.likelihood.eval()

        pred = self.gp_model.likelihood(self.gp_model(candidate_set))

        mu = pred.mean().detach()
        sigma = pred.std().detach()

        u = (self.best_y - mu) / sigma
        m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        ucdf = m.cdf(u)
        updf = torch.exp(m.log_prob(u))
        ei = sigma * (updf + u * ucdf)

        return ei
