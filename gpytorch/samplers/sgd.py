import torch
import warnings
import math
import numpy as np
import copy

from .sgld import SGLD

def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)

class SGD:
    def __init__(self, parameters, model_ell_func, n_samples, method = 'Adam', lr=1e-2, **kwargs):
        self.f_init = flatten(parameters)
        # print("FINIT")
        # print(self.f_init)
        # print("\n\n")

        self.n = self.f_init.nelement()
        if len(self.f_init.size()) < 2:
            self.f_init = self.f_init.unsqueeze(1)
        self.model_ell_func = model_ell_func
        self.n_samples = n_samples
        self.kwargs = kwargs

        self.parameters = parameters
        # print(parameters)
        if method == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters, amsgrad = True, lr = lr)
        elif method == 'SGLD':
            self.optimizer = SGLD(self.parameters, lr = lr, noise_factor=2e-5)

    def run(self):
        self.f_sampled = torch.zeros(self.n, self.n_samples)
        self.ell = torch.zeros(self.n_samples, 1)

        for ii in range(self.n_samples):
            torch.cuda.empty_cache()
            # zero gradients
            self.optimizer.zero_grad()

            # compute negative log likelihood
            ell_cur = -self.model_ell_func(self.parameters, **self.kwargs)

            # take optimization steps
            ell_cur.backward()
            self.optimizer.step()

            # store parameters
            with torch.no_grad():
                self.ell[ii,0] = ell_cur.detach().item()
                output_pars = flatten(self.parameters).detach().view(-1)
                self.f_sampled[:, ii] = output_pars

            #print('ell = %s' % -self.ell[ii,0])

        return self.f_sampled, self.ell
