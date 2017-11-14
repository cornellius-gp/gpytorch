import torch
import gpytorch
from gpytorch.lazy import LazyVariable
from .posterior_strategy import PosteriorStrategy


class DefaultPosteriorStrategy(PosteriorStrategy):
    def alpha_size(self):
        return self.var.size()[1]

    def exact_posterior_alpha(self, train_mean, train_y):
        res = gpytorch.inv_matmul(self.var, train_y - train_mean)
        return res

    def exact_posterior_mean(self, test_mean, alpha):
        if isinstance(self.var, LazyVariable):
            return self.var.matmul(alpha) + test_mean
        return torch.addmv(test_mean, self.var, alpha)
