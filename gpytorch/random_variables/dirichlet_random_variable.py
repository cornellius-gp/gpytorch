from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
from torch.autograd import Variable
from .random_variable import RandomVariable


class DirichletRandomVariable(RandomVariable):
    def __init__(self, alpha):
        """
        Constructs a Dirichlet random variable
        alpha represents the weights of the distribution

        Passing a vector alpha corresponds to a single Dirichlet variable
        Passing a matrix alpha corresponds to a batch of independent Dirichlet variables.

        Params:
        - alpha (Variable: vector k or matrix n x k) weights of categorical distribution
        """
        super(DirichletRandomVariable, self).__init__(alpha)
        if not isinstance(alpha, Variable):
            raise RuntimeError("alpha should be a Variable")

        ndimension = alpha.ndimension()
        if ndimension not in [1, 2]:
            raise RuntimeError("alpha should be a vector or a matrix")
        self.alpha = alpha

    def representation(self):
        return self.alpha

    def sample(self, n_samples=1):
        ndimension = self.alpha.ndimension()
        alpha = self.alpha.data
        if ndimension == 1:
            alpha = alpha.unsqueeze(0)

        batch_size, n_categories = alpha.size()
        res = alpha.new().resize_(n_samples, batch_size, n_categories).zero_()
        for i in range(batch_size):
            np_sample = np.random.dirichlet(alpha[i].cpu().numpy(), size=n_samples)
            res[:, i, :].copy_(torch.from_numpy(np_sample).type_as(res))

        if ndimension == 1:
            res.squeeze_(1)
        return res.type_as(self.alpha.data)
