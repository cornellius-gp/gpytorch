from ..lazy import SumLazyVariable
from ..random_variables import GaussianRandomVariable


class SumVariationalStrategy(object):
    def __init__(self, *variational_strategies):
        self.variational_strategies = variational_strategies

    def mvn_kl_divergence(self):
        return sum(strategy.mvn_kl_divergence() for strategy in self.variational_strategies)

    def variational_samples(self, output, n_samples=None):
        if not isinstance(output.covar(), SumLazyVariable):
            raise RuntimeError
        output_components = [GaussianRandomVariable(output.mean(), covar_component)
                             for covar_component in output.covar().lazy_vars]
        return sum(strategy.variational_samples(output_component)
                   for strategy, output_component in zip(self.variational_strategies, output_components))
