#!/usr/bin/env python3

import torch
from torch.distributions.kl import kl_divergence

from ..distributions import Delta, MultivariateNormal
from ..lazy import MatmulLazyTensor, SumLazyTensor
from ..utils.errors import CachingError
from ..utils.memoize import pop_from_cache_ignore_args
from .delta_variational_distribution import DeltaVariationalDistribution
from .variational_strategy import VariationalStrategy


class BatchDecoupledVariationalStrategy(VariationalStrategy):
    r"""
    A VariationalStrategy that uses a different set of inducing points for the
    variational mean and variational covar.  It follows the "decoupled" model
    proposed by `Jankowiak et al. (2020)`_ (which is roughly based on the strategies
    proposed by `Cheng et al. (2017)`_.

    Let :math:`\mathbf Z_\mu` and :math:`\mathbf Z_\sigma` be the mean/variance
    inducing points. The variational distribution for an input :math:`\mathbf
    x` is given by:

    .. math::

        \begin{align*}
            \mathbb E[ f(\mathbf x) ] &= \mathbf k_{\mathbf Z_\mu \mathbf x}^\top
            \mathbf K_{\mathbf Z_\mu \mathbf Z_\mu}^{-1} \mathbf m
            \\
            \text{Var}[ f(\mathbf x) ] &= k_{\mathbf x \mathbf x} - \mathbf k_{\mathbf Z_\sigma \mathbf x}^\top
            \mathbf K_{\mathbf Z_\sigma \mathbf Z_\sigma}^{-1}
            \left( \mathbf K_{\mathbf Z_\sigma} - \mathbf S \right)
            \mathbf K_{\mathbf Z_\sigma \mathbf Z_\sigma}^{-1}
            \mathbf k_{\mathbf Z_\sigma \mathbf x}
        \end{align*}

    where :math:`\mathbf m` and :math:`\mathbf S` are the variational parameters.
    Unlike the original proposed implementation, :math:`\mathbf Z_\mu` and :math:`\mathbf Z_\sigma`
    have **the same number of inducing points**, which allows us to perform batched operations.

    Additionally, you can use a different set of kernel hyperparameters for the mean and the variance function.
    We recommend using this feature only with the :obj:`~gpytorch.mlls.PredictiveLogLikelihood` objective function
    as proposed in "Parametric Gaussian Process Regressors" (`Jankowiak et al. (2020)`_).
    Use the :attr:`mean_var_batch_dim` to indicate which batch dimension corresponds to the different mean/var
    kernels.

    .. note::
        We recommend using the "right-most" batch dimension (i.e. :attr:`mean_var_batch_dim=-1`) for the dimension
        that corresponds to the different mean/variance kernel parameters.

        Assuming you want `b1` many independent GPs, the :obj:`~gpytorch.variational._VariationalDistribution`
        objects should have a batch shape of `b1`, and the mean/covar modules
        of the GP should have a batch shape of `b1 x 2`.
        (The 2 corresponds to the mean/variance hyperparameters.)

    .. seealso::
        :obj:`~gpytorch.variational.OrthogonallyDecoupledVariationalStrategy` (a variant proposed by
        `Salimbeni et al. (2018)`_ that uses orthogonal projections.)

    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param torch.Tensor inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :type learn_inducing_locations: `bool`, optional
    :type mean_var_batch_dim: `int`, optional
    :param mean_var_batch_dim: (Default `None`):
        Set this parameter (ideally to `-1`) to indicate which dimension corresponds to different
        kernel hyperparameters for the mean/variance functions.

    .. _Cheng et al. (2017):
        https://arxiv.org/abs/1711.10127

    .. _Salimbeni et al. (2018):
        https://arxiv.org/abs/1809.08820

    .. _Jankowiak et al. (2020):
        https://arxiv.org/abs/1910.07123

    Example (**different** hypers for mean/variance):
        >>> class MeanFieldDecoupledModel(gpytorch.models.ApproximateGP):
        >>>     '''
        >>>     A batch of 3 independent MeanFieldDecoupled PPGPR models.
        >>>     '''
        >>>     def __init__(self, inducing_points):
        >>>         # The variational parameters have a batch_shape of [3]
        >>>         variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
        >>>             inducing_points.size(-1), batch_shape=torch.Size([3]),
        >>>         )
        >>>         variational_strategy = gpytorch.variational.BatchDecoupledVariationalStrategy(
        >>>             self, inducing_points, variational_distribution, learn_inducing_locations=True,
        >>>             mean_var_batch_dim=-1
        >>>         )
        >>>
        >>>         # The mean/covar modules have a batch_shape of [3, 2]
        >>>         # where the last batch dim corresponds to the mean & variance hyperparameters
        >>>         super().__init__(variational_strategy)
        >>>         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([3, 2]))
        >>>         self.covar_module = gpytorch.kernels.ScaleKernel(
        >>>             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([3, 2])),
        >>>             batch_shape=torch.Size([3, 2]),
        >>>         )

    Example (**shared** hypers for mean/variance):
        >>> class MeanFieldDecoupledModel(gpytorch.models.ApproximateGP):
        >>>     '''
        >>>     A batch of 3 independent MeanFieldDecoupled PPGPR models.
        >>>     '''
        >>>     def __init__(self, inducing_points):
        >>>         # The variational parameters have a batch_shape of [3]
        >>>         variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
        >>>             inducing_points.size(-1), batch_shape=torch.Size([3]),
        >>>         )
        >>>         variational_strategy = gpytorch.variational.BatchDecoupledVariationalStrategy(
        >>>             self, inducing_points, variational_distribution, learn_inducing_locations=True,
        >>>         )
        >>>
        >>>         # The mean/covar modules have a batch_shape of [3, 1]
        >>>         # where the singleton dimension corresponds to the shared mean/variance hyperparameters
        >>>         super().__init__(variational_strategy)
        >>>         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([3, 1]))
        >>>         self.covar_module = gpytorch.kernels.ScaleKernel(
        >>>             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([3, 1])),
        >>>             batch_shape=torch.Size([3, 1]),
        >>>         )
    """

    def __init__(
        self, model, inducing_points, variational_distribution, learn_inducing_locations=True, mean_var_batch_dim=None
    ):
        if isinstance(variational_distribution, DeltaVariationalDistribution):
            raise NotImplementedError(
                "BatchDecoupledVariationalStrategy does not work with DeltaVariationalDistribution"
            )

        if mean_var_batch_dim is not None and mean_var_batch_dim >= 0:
            raise ValueError(f"mean_var_batch_dim should be negative indexed, got {mean_var_batch_dim}")
        self.mean_var_batch_dim = mean_var_batch_dim

        # Maybe unsqueeze inducing points
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        # We're going to create two set of inducing points
        # One set for computing the mean, one set for computing the variance
        if self.mean_var_batch_dim is not None:
            inducing_points = torch.stack([inducing_points, inducing_points], dim=(self.mean_var_batch_dim - 2))
        else:
            inducing_points = torch.stack([inducing_points, inducing_points], dim=-3)
        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations)

    def _expand_inputs(self, x, inducing_points):
        # If we haven't explicitly marked a dimension as batch, add the corresponding batch dimension to the input
        if self.mean_var_batch_dim is None:
            x = x.unsqueeze(-3)
        else:
            x = x.unsqueeze(self.mean_var_batch_dim - 2)
        return super()._expand_inputs(x, inducing_points)

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        # We'll compute the covariance, and cross-covariance terms for both the
        # pred-mean and pred-covar, using their different inducing points (and maybe kernel hypers)

        mean_var_batch_dim = self.mean_var_batch_dim or -1

        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook to make this cleaner
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.inv_matmul(induc_data_covar.double()).to(full_inputs.dtype)
        mean_interp_term = interp_term.select(mean_var_batch_dim - 2, 0)
        var_interp_term = interp_term.select(mean_var_batch_dim - 2, 1)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} m + \mu_X
        # Here we're using the terms that correspond to the mean's inducing points
        predictive_mean = torch.add(
            torch.matmul(mean_interp_term.transpose(-1, -2), inducing_values.unsqueeze(-1)).squeeze(-1),
            test_mean.select(mean_var_batch_dim - 1, 0),
        )

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLazyTensor(variational_inducing_covar, middle_term)
        predictive_covar = SumLazyTensor(
            data_data_covar.add_jitter(1e-4).evaluate().select(mean_var_batch_dim - 2, 1),
            MatmulLazyTensor(var_interp_term.transpose(-1, -2), middle_term @ var_interp_term),
        )

        return MultivariateNormal(predictive_mean, predictive_covar)

    def kl_divergence(self):
        variational_dist = self.variational_distribution
        prior_dist = self.prior_distribution

        mean_dist = Delta(variational_dist.mean)
        covar_dist = MultivariateNormal(
            torch.zeros_like(variational_dist.mean), variational_dist.lazy_covariance_matrix
        )
        return kl_divergence(mean_dist, prior_dist) + kl_divergence(covar_dist, prior_dist)
