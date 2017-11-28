import gpytorch
import torch
from torch import nn
from gpytorch.random_variables import GaussianRandomVariable, CategoricalRandomVariable
from .likelihood import Likelihood


class SoftmaxLikelihood(Likelihood):
    """
    Implements the Softmax (multiclass) likelihood used for GP classification.
    """
    def __init__(self, n_features, n_classes):
        super(SoftmaxLikelihood, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes

        mixing_weights = nn.Parameter(torch.ones(n_classes, n_features).fill_(1. / n_features))
        self.register_parameter('mixing_weights', mixing_weights, bounds=(-2, 2))

    def forward(self, latent_func):
        """
        Computes predictive distributions p(y|x) given a latent distribution
        p(f|x). To do this, we solve the integral:

            p(y|x) = \int p(y|f)p(f|x) df

        Given that p(y=1|f) = \Phi(f), this integral is analytically tractable,
        and if \mu_f and \sigma^2_f are the mean and variance of p(f|x), the
        solution is given by:

            p(y|x) = \Phi(\frac{\mu}{\sqrt{1+\sigma^2_f}})
        """
        if not isinstance(latent_func, GaussianRandomVariable):
            raise RuntimeError(' '.join([
                'SoftmaxLikelihood expects a Gaussian',
                'distributed latent function to make predictions',
            ]))

        n_samples = gpytorch.functions.num_samples
        samples = latent_func.sample(n_samples)
        if samples.ndimension() != 3:
            raise RuntimeError('f should have 3 dimensions: features x data x samples')
        n_features, n_data, _ = samples.size()
        if n_features != self.n_features:
            raise RuntimeError('There should be %d features' % self.n_features)

        mixed_fs = self.mixing_weights.matmul(samples.view(n_features, n_samples * n_data))
        softmax = nn.functional.softmax(mixed_fs.t()).view(n_data, n_samples, self.n_classes)
        softmax = softmax.mean(1)
        return CategoricalRandomVariable(softmax)

    def log_probability(self, latent_func, target):
        """
        Computes the log probability \sum_{i} \log \Phi(y_{i}f_{i}), where
        \Phi(y_{i}f_{i}) is computed by averaging over a set of s samples of
        f_{i} drawn from p(f|x).
        """
        n_samples = gpytorch.functions.num_samples
        samples = latent_func.sample(n_samples)
        if samples.ndimension() != 3:
            raise RuntimeError('f should have 3 dimensions: features x data x samples')
        n_features, n_data, _ = samples.size()
        if n_features != self.n_features:
            raise RuntimeError('There should be %d features' % self.n_features)

        mixed_fs = self.mixing_weights.matmul(samples.view(n_features, n_samples * n_data))
        log_prob = -nn.functional.cross_entropy(mixed_fs.t(), target.unsqueeze(1).repeat(1, n_samples).view(-1),
                                                size_average=False)
        return log_prob.div(n_samples)
