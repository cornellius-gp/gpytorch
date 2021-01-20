#!/usr/bin/env python3

import logging
import math
from typing import Tuple, Union

import torch

from ..constraints import Interval, Positive
from ..priors import Prior
from .kernel import Kernel

logger = logging.getLogger()


class SpectralMixtureKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Spectral Mixture Kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.

    It was proposed in `Gaussian Process Kernels for Pattern Discovery and Extrapolation`_.

    .. note::
        Unlike other kernels,

            * :attr:`ard_num_dims` **must equal** the number of dimensions of the data.
            * This kernel should not be combined with a :class:`gpytorch.kernels.ScaleKernel`.

    :param int num_mixtures: The number of components in the mixture.
    :param int ard_num_dims: Set this to match the dimensionality of the input.
        It should be `d` if :attr:`x1` is a `... x n x d` matrix. (Default: `1`.)
    :param batch_shape: Set this if the data is batch of input data. It should
        be `b_1 x ... x b_j` if :attr:`x1` is a `b_1 x ... x b_j x n x d` tensor. (Default: `torch.Size([])`.)
    :type batch_shape: torch.Size, optional
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the dimensions. (Default: `None`.)
    :type active_dims: float, optional
    :param eps: The minimum value that the lengthscale can take (prevents divide by zero errors). (Default: `1e-6`.)
    :type eps: float, optional

    :param mixture_scales_prior: A prior to set on the :attr:`mixture_scales` parameter
    :type mixture_scales_prior: ~gpytorch.priors.Prior, optional
    :param mixture_scales_constraint: A constraint to set on the :attr:`mixture_scales` parameter
    :type mixture_scales_constraint: ~gpytorch.constraints.Interval, optional
    :param mixture_means_prior: A prior to set on the :attr:`mixture_means` parameter
    :type mixture_means_prior: ~gpytorch.priors.Prior, optional
    :param mixture_means_constraint: A constraint to set on the :attr:`mixture_means` parameter
    :type mixture_means_constraint: ~gpytorch.constraints.Interval, optional
    :param mixture_weights_prior: A prior to set on the :attr:`mixture_weights` parameter
    :type mixture_weights_prior: ~gpytorch.priors.Prior, optional
    :param mixture_weights_constraint: A constraint to set on the :attr:`mixture_weights` parameter
    :type mixture_weights_constraint: ~gpytorch.constraints.Interval, optional

    :ivar torch.Tensor mixture_scales: The lengthscale parameter. Given
        `k` mixture components, and `... x n x d` data, this will be of size `... x k x 1 x d`.
    :ivar torch.Tensor mixture_means: The mixture mean parameters (`... x k x 1 x d`).
    :ivar torch.Tensor mixture_weights: The mixture weight parameters (`... x k`).

    Example:

        >>> # Non-batch
        >>> x = torch.randn(10, 5)
        >>> covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=5)
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)
        >>>
        >>> # Batch
        >>> batch_x = torch.randn(2, 10, 5)
        >>> covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, batch_size=2, ard_num_dims=5)
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)

    .. _Gaussian Process Kernels for Pattern Discovery and Extrapolation:
        https://arxiv.org/pdf/1302.4245.pdf
    """

    is_stationary = True  # kernel is stationary even though it does not have a lengthscale

    def __init__(
        self,
        num_mixtures: int = None,
        ard_num_dims: int = 1,
        batch_shape: torch.Size = torch.Size([]),
        mixture_scales_prior: Prior = None,
        mixture_scales_constraint: Interval = None,
        mixture_means_prior: Prior = None,
        mixture_means_constraint: Interval = None,
        mixture_weights_prior: Prior = None,
        mixture_weights_constraint: Interval = None,
        **kwargs,
    ):
        if num_mixtures is None:
            raise RuntimeError("num_mixtures is a required argument")
        if mixture_means_prior is not None or mixture_scales_prior is not None or mixture_weights_prior is not None:
            logger.warning("Priors not implemented for SpectralMixtureKernel")

        # This kernel does not use the default lengthscale
        super(SpectralMixtureKernel, self).__init__(ard_num_dims=ard_num_dims, batch_shape=batch_shape, **kwargs)
        self.num_mixtures = num_mixtures

        if mixture_scales_constraint is None:
            mixture_scales_constraint = Positive()

        if mixture_means_constraint is None:
            mixture_means_constraint = Positive()

        if mixture_weights_constraint is None:
            mixture_weights_constraint = Positive()

        self.register_parameter(
            name="raw_mixture_weights", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.num_mixtures))
        )
        ms_shape = torch.Size([*self.batch_shape, self.num_mixtures, 1, self.ard_num_dims])
        self.register_parameter(name="raw_mixture_means", parameter=torch.nn.Parameter(torch.zeros(ms_shape)))
        self.register_parameter(name="raw_mixture_scales", parameter=torch.nn.Parameter(torch.zeros(ms_shape)))

        self.register_constraint("raw_mixture_scales", mixture_scales_constraint)
        self.register_constraint("raw_mixture_means", mixture_means_constraint)
        self.register_constraint("raw_mixture_weights", mixture_weights_constraint)

    @property
    def mixture_scales(self):
        return self.raw_mixture_scales_constraint.transform(self.raw_mixture_scales)

    @mixture_scales.setter
    def mixture_scales(self, value: Union[torch.Tensor, float]):
        self._set_mixture_scales(value)

    def _set_mixture_scales(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_scales)
        self.initialize(raw_mixture_scales=self.raw_mixture_scales_constraint.inverse_transform(value))

    @property
    def mixture_means(self):
        return self.raw_mixture_means_constraint.transform(self.raw_mixture_means)

    @mixture_means.setter
    def mixture_means(self, value: Union[torch.Tensor, float]):
        self._set_mixture_means(value)

    def _set_mixture_means(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_means)
        self.initialize(raw_mixture_means=self.raw_mixture_means_constraint.inverse_transform(value))

    @property
    def mixture_weights(self):
        return self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)

    @mixture_weights.setter
    def mixture_weights(self, value: Union[torch.Tensor, float]):
        self._set_mixture_weights(value)

    def _set_mixture_weights(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_weights)
        self.initialize(raw_mixture_weights=self.raw_mixture_weights_constraint.inverse_transform(value))

    def initialize_from_data_empspect(self, train_x: torch.Tensor, train_y: torch.Tensor):
        """
        Initialize mixture components based on the empirical spectrum of the data.
        This will often be better than the standard initialize_from_data method, but it assumes
        that your inputs are evenly spaced.

        :param torch.Tensor train_x: Training inputs
        :param torch.Tensor train_y: Training outputs
        """

        import numpy as np
        from scipy.fftpack import fft
        from scipy.integrate import cumtrapz

        with torch.no_grad():
            if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
                raise RuntimeError("train_x and train_y should be tensors")
            if train_x.ndimension() == 1:
                train_x = train_x.unsqueeze(-1)
            if self.active_dims is not None:
                train_x = train_x[..., self.active_dims]

            # Flatten batch dimensions
            train_x = train_x.view(-1, train_x.size(-1))
            train_y = train_y.view(-1)

            N = train_x.size(-2)
            emp_spect = np.abs(fft(train_y.cpu().detach().numpy())) ** 2 / N
            M = math.floor(N / 2)

            freq1 = np.arange(M + 1)
            freq2 = np.arange(-M + 1, 0)
            freq = np.hstack((freq1, freq2)) / N
            freq = freq[: M + 1]
            emp_spect = emp_spect[: M + 1]

            total_area = np.trapz(emp_spect, freq)
            spec_cdf = np.hstack((np.zeros(1), cumtrapz(emp_spect, freq)))
            spec_cdf = spec_cdf / total_area

            a = np.random.rand(1000, self.ard_num_dims)
            p, q = np.histogram(a, spec_cdf)
            bins = np.digitize(a, q)
            slopes = (spec_cdf[bins] - spec_cdf[bins - 1]) / (freq[bins] - freq[bins - 1])
            intercepts = spec_cdf[bins - 1] - slopes * freq[bins - 1]
            inv_spec = (a - intercepts) / slopes

            from sklearn.mixture import GaussianMixture

            GMM = GaussianMixture(n_components=self.num_mixtures, covariance_type="diag").fit(inv_spec)
            means = GMM.means_
            varz = GMM.covariances_
            weights = GMM.weights_

            dtype = self.raw_mixture_means.dtype
            device = self.raw_mixture_means.device
            self.mixture_means = torch.tensor(means, dtype=dtype, device=device).unsqueeze(-2)
            self.mixture_scales = torch.tensor(varz, dtype=dtype, device=device).unsqueeze(-2)
            self.mixture_weights = torch.tensor(weights, dtype=dtype, device=device)

    def initialize_from_data(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs):
        """
        Initialize mixture components based on batch statistics of the data. You should use
        this initialization routine if your observations are not evenly spaced.

        :param torch.Tensor train_x: Training inputs
        :param torch.Tensor train_y: Training outputs
        """

        with torch.no_grad():
            if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
                raise RuntimeError("train_x and train_y should be tensors")
            if train_x.ndimension() == 1:
                train_x = train_x.unsqueeze(-1)
            if self.active_dims is not None:
                train_x = train_x[..., self.active_dims]

            # Compute maximum distance between points in each dimension
            train_x_sort = train_x.sort(dim=-2)[0]
            max_dist = train_x_sort[..., -1, :] - train_x_sort[..., 0, :]

            # Compute the minimum distance between points in each dimension
            dists = train_x_sort[..., 1:, :] - train_x_sort[..., :-1, :]
            # We don't want the minimum distance to be zero, so fill zero values with some large number
            dists = torch.where(dists.eq(0.0), torch.tensor(1.0e10, dtype=train_x.dtype, device=train_x.device), dists)
            sorted_dists = dists.sort(dim=-2)[0]
            min_dist = sorted_dists[..., 0, :]

            # Reshape min_dist and max_dist to match the shape of parameters
            # First add a singleton data dimension (-2) and a dimension for the mixture components (-3)
            min_dist = min_dist.unsqueeze_(-2).unsqueeze_(-3)
            max_dist = max_dist.unsqueeze_(-2).unsqueeze_(-3)
            # Compress any dimensions in min_dist/max_dist that correspond to singletons in the SM parameters
            dim = -3
            while -dim <= min_dist.dim():
                if -dim > self.raw_mixture_scales.dim():
                    min_dist = min_dist.min(dim=dim)[0]
                    max_dist = max_dist.max(dim=dim)[0]
                elif self.raw_mixture_scales.size(dim) == 1:
                    min_dist = min_dist.min(dim=dim, keepdim=True)[0]
                    max_dist = max_dist.max(dim=dim, keepdim=True)[0]
                    dim -= 1
                else:
                    dim -= 1

            # Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2) |
            self.mixture_scales = torch.randn_like(self.raw_mixture_scales).mul_(max_dist).abs_().reciprocal_()
            # Draw means from Unif(0, 0.5 / minimum distance between two points)
            self.mixture_means = torch.rand_like(self.raw_mixture_means).mul_(0.5).div(min_dist)
            # Mixture weights should be roughly the stdv of the y values divided by the number of mixtures
            self.mixture_weights = train_y.std().div(self.num_mixtures)

    def _create_input_grid(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a helper method for creating a grid of the kernel's inputs.
        Use this helper rather than maually creating a meshgrid.

        The grid dimensions depend on the kernel's evaluation mode.

        :param torch.Tensor x1: ... x n x d
        :param torch.Tensor x2: ... x m x d (for diag mode, these must be the same inputs)
        :param diag: Should the Kernel compute the whole kernel, or just the diag? (Default: True.)
        :type diag: bool, optional
        :param last_dim_is_batch: If this is true, it treats the last dimension
            of the data as another batch dimension.  (Useful for additive
            structure over the dimensions). (Default: False.)
        :type last_dim_is_batch: bool, optional

        :rtype: torch.Tensor, torch.Tensor
        :return: Grid corresponding to x1 and x2. The shape depends on the kernel's mode:
            * `full_covar`: (`... x n x 1 x d` and `... x 1 x m x d`)
            * `full_covar` with `last_dim_is_batch=True`: (`... x k x n x 1 x 1` and `... x k x 1 x m x 1`)
            * `diag`: (`... x n x d` and `... x n x d`)
            * `diag` with `last_dim_is_batch=True`: (`... x k x n x 1` and `... x k x n x 1`)
        """
        x1_, x2_ = x1, x2
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
            if torch.equal(x1, x2):
                x2_ = x1_
            else:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

        if diag:
            return x1_, x2_
        else:
            return x1_.unsqueeze(-2), x2_.unsqueeze(-3)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, num_dims = x1.shape[-2:]

        if not num_dims == self.ard_num_dims:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have {} dimensionality "
                "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, num_dims)
            )

        # Expand x1 and x2 to account for the number of mixtures
        # Should make x1/x2 (... x k x n x d) for k mixtures
        x1_ = x1.unsqueeze(-3)
        x2_ = x2.unsqueeze(-3)

        # Compute distances - scaled by appropriate parameters
        x1_exp = x1_ * self.mixture_scales
        x2_exp = x2_ * self.mixture_scales
        x1_cos = x1_ * self.mixture_means
        x2_cos = x2_ * self.mixture_means

        # Create grids
        x1_exp_, x2_exp_ = self._create_input_grid(x1_exp, x2_exp, diag=diag, **params)
        x1_cos_, x2_cos_ = self._create_input_grid(x1_cos, x2_cos, diag=diag, **params)

        # Compute the exponential and cosine terms
        exp_term = (x1_exp_ - x2_exp_).pow_(2).mul_(-2 * math.pi ** 2)
        cos_term = (x1_cos_ - x2_cos_).mul_(2 * math.pi)
        res = exp_term.exp_() * cos_term.cos_()

        # Sum over mixtures
        mixture_weights = self.mixture_weights.view(*self.mixture_weights.shape, 1, 1)
        if not diag:
            mixture_weights = mixture_weights.unsqueeze(-2)

        res = (res * mixture_weights).sum(-3 if diag else -4)

        # Product over dimensions
        if last_dim_is_batch:
            # Put feature-dimension in front of data1/data2 dimensions
            res = res.permute(*list(range(0, res.dim() - 3)), -1, -3, -2)
        else:
            res = res.prod(-1)

        return res
