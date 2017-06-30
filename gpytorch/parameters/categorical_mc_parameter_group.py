import torch
from .mc_parameter_group import MCParameterGroup
from ..random_variables import CategoricalRandomVariable, IndependentRandomVariables

class CategoricalMCParameterGroup(MCParameterGroup):
    def __init__():
        self._priors = {}
        self._options = {'num_samples': 20}

        for name, prior in kwargs.items():
            if isinstance(prior,IndependentRandomVariables) and not all([isinstance(sub_prior, CategoricalRandomVariable) for sub_prior in prior]):
                raise RuntimeError('All priors over a single parameter must be CategoricalRandomVariables')

            if not isinstance(prior, CategoricalRandomVariable):
                raise RuntimeError('All parameters in an MCParameterGroup must have priors of type CategoricalRandomVariable')

            self._update_buffer[name] = Variable(torch.zeros(len(prior)))
            self._priors[name] = prior
            self._samples[name] = torch.Tensor()

            self.register


    def sample(self):
        if not self._training:
            ix = random.randint(0,self._options['num_samples'])
            for param_name, param_value in self:
                if len(self._samples[param_name]) > 0:
                    param_value.data = self._samples[param_name][ix]
                else:
                    for i in range(len(param_value)):
                        param_value.data[i] = self._priors[param_name].sample()


    def update(self, log_likelihood_closure):
        # Reset sample sets, set initial samples
        for param_name in self:
            if len(self._samples[param_name]) > 0:
                self._samples[param_name] = [self._samples[param_name][-1]]
                setattr(self,param_name,self._samples[param_name][0])
            else:
                setattr(self,param_name,self._priors[param_name].sample())

        # Draw samples
        for i in range(self._options['num_samples']):
            # Do a single round of Gibbs sampling for each parameter in turn

            for param_name, param_value in self:
                self._samples[param_name][i] = torch.zeros(len(param_value))
                prior = self._priors[param_name]
                num_categories = prior.num_categories()

                # Do Gibbs sampling for the elements of this parameter
                for j in range(len(param_value)):
                    log_posts = torch.zeros(num_categories)

                    # get log posteriors for each possible category
                    for k in range(num_categories):
                        param_value.data[j] = k
                        log_posts[k] = log_likelihood_closure() + prior.log_probability(k)

                    # get posterior probabilities
                    posts = log_posts.exp().div_(log_posts.sum())

                    # Sample from posterior, set parameter value
                    post_sample = CategoricalPrior(posts).sample()

                    self._samples[param_name][i][j] = post_sample
                    param_value.data[j] = post_sample






