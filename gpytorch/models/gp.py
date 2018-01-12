from ..module import Module


class GP(Module):
    def marginal_log_likelihood(self, likelihood, output, target, n_data=None):
        """
        Returns the marginal log likelihood of the data

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - output: (GaussianRandomVariable) - the output of the GP model
        - target: (Variable) - target
        - n_data: (int) - total number of data points in the set (required only for SGD)
        """
        raise NotImplementedError
