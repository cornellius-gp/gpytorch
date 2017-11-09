from copy import deepcopy
import gpytorch
from torch.autograd import Variable
from .variational_strategy import VariationalStrategy
from ..lazy import KroneckerProductLazyVariable


class GridInducingPointStrategy(VariationalStrategy):
    def variational_samples(self, output, n_samples=None):
        if n_samples is None:
            n_samples = gpytorch.functions.num_trace_samples

        # Draw samplse from variational distribution
        base_samples = Variable(self.variational_mean.data.new(len(self.variational_mean), n_samples).normal_())
        samples = self.chol_variational_covar.t().mm(base_samples)
        samples = samples + self.variational_mean.unsqueeze(1)

        # Hacky code for now for KroneckerProductLazyVariable. Let's change it soon.
        if isinstance(output.covar(), KroneckerProductLazyVariable):
            interp_matrix = output.covar().representation()[1]
            samples = gpytorch.dsmm(interp_matrix, samples)
            return samples

        interp_indices = Variable(output.covar().J_left)
        interp_values = Variable(output.covar().C_left)
        # Left multiply samples by interpolation matrix
        interp_size = list(interp_indices.size()) + [samples.size(-1)]
        samples_size = deepcopy(interp_size)
        samples_size[-3] = samples.size()[-2]
        interp_indices_expanded = interp_indices.unsqueeze(-1).expand(*interp_size)
        samples_output = samples.unsqueeze(-2).expand(*samples_size).gather(-3, interp_indices_expanded)
        samples_output = samples_output.mul(interp_values.unsqueeze(-1).expand(interp_size))
        samples = samples_output.sum(-2)

        return samples
