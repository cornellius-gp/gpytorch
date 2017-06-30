import torch
import random
from collections import namedtuple
from .mc_parameter_group import MCParameterGroup
from ..random_variables import CategoricalRandomVariable, BatchRandomVariables

class CategoricalMCParameterGroup(MCParameterGroup):
    def __init__(self, **kwargs):
        super(CategoricalMCParameterGroup,self).__init__()
        for name, prior in kwargs.items():
            if isinstance(prior,BatchRandomVariables) and not all([isinstance(sub_prior, CategoricalRandomVariable) for sub_prior in prior]):
                raise RuntimeError('All priors over a single parameter must be CategoricalRandomVariables')

            if not isinstance(prior, CategoricalRandomVariable):
                raise RuntimeError('All parameters in an MCParameterGroup must have priors of type CategoricalRandomVariable')

            self._update_buffer[name] = Variable(torch.zeros(len(prior)))
            self._priors[name] = prior

    def update(self, log_likelihood_closure):
        self._training = False

        for name, rv in self:
            self._update_buffer[name] = ConstantRandomVariable(rv.sample())

        self._training = True

        for name, prior in self._priors.items():
            if name not in self._posterior.keys():
                size = (num_samples,) if isinstance(prior, CategoricalRandomVariable) else (num_samples, len(prior))
                sample_buffer = torch.zeros(*size)
                self.register_buffer('%s_samples' % name, sample_buffer)
                self._posterior[name] = SamplesRandomVariable(sample_buffer)

        # Draw samples
        for i in range(self._options['num_samples'])
            # Do a single round of Gibbs sampling for each parameter in turn
            for name, prior in self._priors.items():
                if isinstance(prior, CategoricalRandomVariable):
                    prior = BatchRandomVariables(prior, 1)

                param_length = len(prior)
                num_categories = prior[0].num_categories()

                # Do Gibbs sampling for the elements of this parameter
                for j in range(param_length):
                    log_posts = torch.zeros(num_categories)

                    # get log posteriors for each possible category
                    for k in range(num_categories):
                        self._update_buffer[name][j] = k
                        log_posts[k] = log_likelihood_closure() + prior.log_probability(k)

                    # get posterior probabilities
                    posts = log_posts.exp().div_(log_posts.sum())

                    # Sample from posterior, set parameter value
                    post_sample = CategoricalRandomVariable(posts).sample()

                    self._update_buffer[name][j] = post_sample
                    if param_length > 1:
                        self._posterior[name][i,j] = post_sample
                    else:
                        self._posterior[name][i] = post_sample

        self._training = False