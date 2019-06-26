#!/usr/bin/env python3

import torch
import pyro
from ..lazy import CholLazyTensor
from .variational_strategy import VariationalStrategy


class PyroVariationalStrategy(VariationalStrategy):
    def _transform_inputs(self, inputs):
        # Note: assumes inducing_points are not shared (e.g., are O_i x m x O_{i-1})
        # If shared, we need to expand and repeat m x d -> O_i x m x O_{i-1}
        # inputs (x) are either n x O_{0} or p x n x O_{i-1}
        inputs = inputs.contiguous()
        if inputs.dim() == 2:  # n x O_{0}, make O_{i} x n x O_{0}
            # Assume new input entirely
            inputs = inputs.unsqueeze(0)
            inputs = inputs.expand(self.model.output_dims, inputs.size(-2), self.model.input_dims)
        elif inputs.dim() == 3:  # p x n x O_{i-1} -> O_{i} x p x n x O_{i-1}
            # Assume batch dim is samples, not output_dim
            inputs = inputs.unsqueeze(0)
            inputs = inputs.expand(self.model.output_dims, inputs.size(1), inputs.size(-2), self.model.input_dims)

        if inputs.dim() == 4:  # Convert O_{i} x p x n x O_{i-1} -> O_{i} x p*n x O_{i-1}
            num_samples = inputs.size(-3)
            inputs = inputs.view(self.model.output_dims, inputs.size(-2) * inputs.size(-3), self.model.input_dims)
        else:
            num_samples = None

        return inputs, num_samples


class PyroExactVariationalStrategy(PyroVariationalStrategy):
    def forward(self, inputs):
        # Goal: return a p x n x O_{i} tensor.
        inputs, num_samples = self._transform_inputs(inputs)

        # prior_distribution is p(u)
        # induc_mean is O_{i} x m
        # induc_induc_covar is O_{i} x m x m
        prior_distribution = self.prior_distribution
        with self.model.output_dim_plate:
            p_u_samples = pyro.sample(self.model.name_prefix + ".inducing_values", prior_distribution)

        p_f_dist = super().forward(inputs)
        means = p_f_dist.mean
        variances = p_f_dist.variance
        p_f_dist = pyro.distributions.Normal(means, variances.sqrt())

        return p_f_dist, p_u_samples


class PyroSamplingVariationalStrategy(PyroVariationalStrategy):
    def forward(self, inputs):
        # Goal: return a p x n x O_{i} tensor.
        inducing_points = self.inducing_points
        num_induc = inducing_points.size(-2)
        minibatch_size = inputs.size(-2)

        inputs, num_samples = self._transform_inputs(inputs)

        full_inputs = torch.cat([inducing_points, inputs], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

        # full mean is now O_{i} x (p*n + m)

        # Mean terms
        induc_mean = full_mean[..., :num_induc]  # O_{i} x m
        test_mean = full_mean[..., num_induc:]  # O_{i} x (p*n)

        # Covariance terms
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_induc_covar = CholLazyTensor(induc_induc_covar.cholesky())

        # induc_induc_covar is K_mm and is O_{i} x m x m

        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()  # O_{i} x m x p*n
        data_data_covar = full_covar[..., num_induc:, num_induc:]  # O_{i} x p*n x p*n

        # prior_distribution is p(u)
        # induc_mean is O_{i} x m
        # induc_induc_covar is O_{i} x m x m
        prior_distribution = full_output.__class__(induc_mean, induc_induc_covar)
        with self.model.output_dim_plate:
            p_u_samples = pyro.sample(self.model.name_prefix + ".inducing_values", prior_distribution)

        solve_result = induc_induc_covar.inv_matmul((p_u_samples - induc_mean).unsqueeze(-1)).squeeze(-1)
        # solve_result is K_uu^{-1}u and is p x O_{i} x m

        # We need to multiply K_ux^{T} by solve_result.
        # K_ux^{T} is either O_{i} x m x n if this is the first layer in the deep GP
        # or it is O_{i} x m x p*n if it is any later layer.

        if num_samples is not None:  # This means we are in a later layer, and K_ux^{T} is O_{i} x m x p*n
            # We need to reshape O_{i} x m x p*n to p x O_{i} x n x m

            # Step 1: Uncoalesce the p*n dimension to be p x n
            induc_data_covar = induc_data_covar.view(
                self.model.output_dims,
                num_induc,
                num_samples,
                minibatch_size,
            )  # induc_data_covar is now O_{i} x m x p x n
            induc_data_covar = induc_data_covar.permute(2, 0, 1, 3)
            # induc_data_covar is now p x O_{i} x n x m

            # K_xx is also a problem, because it is O_{i} x pn x pn
            # data_data_covar is O_{i} x pn x pn
            data_data_diag = data_data_covar.diag().view(
                self.model.output_dims, solve_result.size(0), minibatch_size
            )

            # diag is O_{i} x p x n
            data_data_diag = data_data_diag.transpose(-3, -2)
            # diag is p x O_{i} x n

            test_mean = test_mean.view(self.model.output_dims, num_samples, minibatch_size).transpose(-3, -2)

        else:  # This is the first layer, and K_ux^{T} is O_{i} x m x n, so there is no p
            # Nothing needs to be done to induc_data_covar
            # And the diagonal of K_xx is just K_xx.diag() (e.g., no p to pull out).
            data_data_diag = data_data_covar.diag()

        means = induc_data_covar.transpose(-2, -1).matmul(solve_result.unsqueeze(-1)).squeeze(-1) + test_mean
        if num_samples is not None:
            # TODO: use inv_quad to compute diag corrections
            diag_correction = (induc_induc_covar.unsqueeze(0).inv_matmul(induc_data_covar) * induc_data_covar).sum(-2)
        else:
            # First layer of deep GP, no need to unsqueeze b/c data only had 2 dims (so induc_data_covar has 3).
            diag_correction = (induc_induc_covar.unsqueeze(0).inv_matmul(induc_data_covar) * induc_data_covar).sum(-2)

        # Computes diag(K_xx) - diag(K_xuK_uu^{-1}K_ux)
        variances = (data_data_diag - diag_correction).clamp_min(0)

        p_f_dist = pyro.distributions.Normal(means, variances.sqrt())

        return p_f_dist, p_u_samples
