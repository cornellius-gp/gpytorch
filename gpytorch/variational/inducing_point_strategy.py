import gpytorch
from torch.autograd import Variable
from .variational_strategy import VariationalStrategy


class InducingPointStrategy(VariationalStrategy):
    def variational_samples(self, output, n_samples=None):
        if n_samples is None:
            n_samples = gpytorch.functions.num_trace_samples

        # Draw samplse from variational distribution
        base_samples = Variable(self.variational_mean.data.new(len(self.variational_mean), n_samples).normal_())
        samples = self.chol_variational_covar.t().mm(base_samples)
        samples = samples + self.variational_mean.unsqueeze(1)
        return samples
