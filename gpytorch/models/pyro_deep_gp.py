import torch
import pyro
from gpytorch.constraints import Positive
from .abstract_variational_gp import AbstractVariationalGP
from ..lazy import CholLazyTensor, DiagLazyTensor
from ..variational import PyroVariationalStrategy, PyroExactVariationalStrategy


class AbstractPyroHiddenGPLayer(AbstractVariationalGP):
    def __init__(self, variational_strategy, input_dims, output_dims, first_layer, name_prefix=""):
        if not isinstance(variational_strategy, PyroVariationalStrategy):
            raise RuntimeError("Pyro GP Layers must have PyroVariationalStrategies!")

        super().__init__(variational_strategy)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.output_dim_plate = pyro.plate(name_prefix + ".n_output_plate", self.output_dims, dim=-1)
        self.name_prefix = name_prefix
        self.first_layer = first_layer

        self.num_inducing = self.variational_strategy.inducing_points.size(-2)

        self.annealing = 1.0

    @property
    def variational_distribution(self):
        return self.variational_strategy.variational_distribution.variational_distribution

    def guide(self):
        with pyro.poutine.scale(scale=self.annealing):
            with self.output_dim_plate:
                q_u_samples = pyro.sample(self.name_prefix + ".inducing_values", self.variational_distribution)
            return q_u_samples

    def model(self, inputs, return_samples=True):
        with pyro.poutine.scale(scale=self.annealing):
            pyro.module(self.name_prefix + ".gp_layer", self)
            # Go from x -> q(f|x) via \int q(f|u)q(u)du, which we evaluate using the variational strategy.
            p_f_dist, p_u_samples = self.variational_strategy(inputs)

            if return_samples:
                # Return appropriately shaped samples from q(f|x)
                if isinstance(self.variational_strategy, PyroExactVariationalStrategy):
                    # In exact mode, because we can't use pyro.sample, we need to sample
                    # num_particles samples in the first layer.
                    sample_shape = (p_u_samples.size(0),) if self.first_layer else ()
                    samples = p_f_dist.rsample(sample_shape=sample_shape).transpose(-2, -1)
                    samples = samples.view(p_u_samples.size(0), -1, self.output_dims)
                else:
                    samples = p_f_dist.rsample().transpose(-2, -1)

                return samples
            else:
                # Return the distribution q(f|x) itself.
                means = p_f_dist.mean
                variances = p_f_dist.variance
                return pyro.distributions.Normal(means.transpose(-2, -1), variances.transpose(-2, -1).sqrt())

    def __call__(self, inputs):
        raise NotImplementedError


class AbstractPyroDeepGP(AbstractPyroHiddenGPLayer):
    def __init__(
        self,
        variational_strategy,
        input_dims,
        output_dims,
        total_num_data,
        hidden_gp_layers,
        likelihood,
        name_prefix="",
    ):
        super().__init__(
            variational_strategy,
            input_dims,
            output_dims,
            first_layer=False,
            name_prefix=name_prefix,
        )

        self.hidden_gp_layers = hidden_gp_layers  # A list of AbstractPyroHiddenGPLayers
        self.total_num_data = total_num_data
        self.likelihood = likelihood
        self.log_beta = torch.nn.Parameter(torch.tensor([3.0]))

    def guide(self, inputs, outputs):
        with pyro.poutine.scale(scale=float(1. / self.total_num_data)):
            for hidden_gp_layer in self.hidden_gp_layers:
                hidden_gp_layer.guide()

            super().guide()

    def model(self, inputs, outputs):
        pyro.param("log_beta", self.log_beta)
        #pyro.module(self.name_prefix + ".likelihood", self.likelihood)
        with pyro.poutine.scale(scale=float(1. / self.total_num_data)):
            pyro.module(self.name_prefix + ".gp_layer", self)
            # First call hidden GP layers
            for hidden_gp_layer in self.hidden_gp_layers:
                inputs = hidden_gp_layer.model(inputs)

            f_samples = super().model(inputs, return_samples=True) #.to_event(1)

            minibatch_size = inputs.size(-2)
            if outputs is not None:
                outputs = outputs.unsqueeze(-1)

            with pyro.plate(self.name_prefix + ".data_plate", minibatch_size, dim=-1):
                with pyro.poutine.scale(scale=float(self.total_num_data / minibatch_size)):
                    #out_dist = QuadratureDist(self.likelihood, p_f_dist)
                    sigma = (-0.5 * self.log_beta).exp()
                    pyro.sample(self.name_prefix + ".output_value",
                                pyro.distributions.Normal(f_samples, sigma).to_event(1), obs=outputs)

            return f_samples

    def __call__(self, inputs, num_samples=10):
        """
        do elegant pyro replay magic
        """
        from pyro.infer.importance import vectorized_importance_weights

        with torch.no_grad():
            _, model_trace, guide_trace = vectorized_importance_weights(self.model, self.guide,
                                                                        inputs, None,
                                                                        num_samples=num_samples,
                                                                        max_plate_nesting=1,
                                                                        normalized=False)
        return(model_trace.nodes['_RETURN']['value'])


