from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .acquisition_function import AcquisitionFunction
from torch.distributions.normal import Normal


class MaxValueEntropySearch(AcquisitionFunction):
    def __init__(self, gp_model, K):
        # K: # of sampled function maxima
        super(MaxValueEntropySearch, self).__init__(gp_model)
        self.K = K

    def forward(self, candidate_set):
        self.gp_model.eval()
        self.gp_model.likelihood.eval()

        pred = self.gp_model.likelihood(self.gp_model(candidate_set))

        mu = pred.mean().detach()
        sigma = pred.std().detach()

        # K samples of the posterior function f
        f_samples = pred.sample(self.K)

        # K samples of y_star
        ys = f_samples.max(dim=0)[0]
        ysArray = ys.unsqueeze(0).expand(candidate_set.shape[0], self.K)

        # compute gamma_y_star
        muArray = mu.unsqueeze(1).expand(candidate_set.shape[0], self.K)
        sigmaArray = sigma.unsqueeze(1).expand(candidate_set.shape[0], self.K)
        gamma = (ysArray - muArray) / sigmaArray

        # Compute the acquisition function of MES.
        m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))  # standard normal
        pdfgamma = torch.exp(m.log_prob(gamma))
        cdfgamma = m.cdf(gamma)

        mve = torch.mean(gamma * pdfgamma / (2 * cdfgamma) - torch.log(cdfgamma), dim=1)
        return mve
