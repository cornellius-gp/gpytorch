import gpytorch
from torch.autograd import Variable
from .variational_strategy import VariationalStrategy
from ..lazy import InterpolatedLazyVariable, KroneckerProductLazyVariable, SumInterpolatedLazyVariable
from ..utils import left_interp


class GridInducingPointStrategy(VariationalStrategy):
    def variational_samples(self, output, n_samples=None):
        if n_samples is None:
            n_samples = gpytorch.functions.num_trace_samples

        # Draw samplse from variational distribution
        base_samples = Variable(self.variational_mean.data.new(self.variational_mean.size(-1), n_samples).normal_())
        if self.variational_mean.ndimension() > 1:
            # Batch mode
            base_samples = base_samples.unsqueeze(0)
        samples = self.chol_variational_covar.transpose(-1, -2).matmul(base_samples)
        samples = samples + self.variational_mean.unsqueeze(-1)

        # Hacky code for now for KroneckerProductLazyVariable. Let's change it soon.
        if isinstance(output.covar(), KroneckerProductLazyVariable):
            interp_matrix = output.covar().representation()[1]
            samples = gpytorch.dsmm(interp_matrix, samples)
            return samples

        if not isinstance(output.covar(), InterpolatedLazyVariable):
            raise RuntimeError('Output should be an interpolated lazy variable')

        # Left multiply samples by interpolation matrix
        interp_indices = output.covar().left_interp_indices
        interp_values = output.covar().left_interp_values

        samples = left_interp(interp_indices, interp_values, samples)
        if isinstance(output.covar(), SumInterpolatedLazyVariable):
            samples = samples.sum(0)

        return samples
