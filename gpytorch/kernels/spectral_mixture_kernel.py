#!/usr/bin/env python3

import logging
import math

import torch

from ..constraints import Positive
from .kernel import Kernel

logger = logging.getLogger()


class SpectralMixtureKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Spectral Mixture Kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:
    It was proposed in `Gaussian Process Kernels for Pattern Discovery and Extrapolation`_.

    .. note::

        Unlike other kernels,
        * :attr:`ard_num_dims` **must equal** the number of dimensions of the data
        * :attr:`batch_shape` **must equal** the batch size of the data (torch.Size([1]) if the data is not batched)
        * :attr:`batch_shape` **cannot** contain more than one batch dimension.
        * This kernel should not be combined with a :class:`gpytorch.kernels.ScaleKernel`.

    Args:
        :attr:`num_mixtures` (int, optional):
            The number of components in the mixture.
        :attr:`ard_num_dims` (int, optional):
            Set this to match the dimensionality of the input.
            It should be `d` if :attr:`x1` is a `n x d` matrix. Default: `1`
        :attr:`batch_shape` (torch.Size, optional):
            Set this if the data is
             batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([1])`
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`mixture_lengthscale` (Tensor):
            The lengthscale parameter. Given `k` mixture components, and `b x n x d` data, this will be of
            size `b x k x 1 x d`.
        :attr:`mixture_means` (Tensor):
            The mixture mean parameters (`b x k x 1 x d`).
        :attr:`mixture_weights` (Tensor):
            The mixture weight parameters (`b x k`).

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
        num_mixtures=None,
        ard_num_dims=1,
        batch_shape=torch.Size([]),
        mixture_scales_prior=None,
        mixture_scales_constraint=None,
        mixture_means_prior=None,
        mixture_means_constraint=None,
        mixture_weights_prior=None,
        mixture_weights_constraint=None,
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
    def mixture_scales(self, value):
        self._set_mixture_scales(value)

    def _set_mixture_scales(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_scales)
        self.initialize(raw_mixture_scales=self.raw_mixture_scales_constraint.inverse_transform(value))

    @property
    def mixture_means(self):
        return self.raw_mixture_means_constraint.transform(self.raw_mixture_means)

    @mixture_means.setter
    def mixture_means(self, value):
        self._set_mixture_means(value)

    def _set_mixture_means(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_means)
        self.initialize(raw_mixture_means=self.raw_mixture_means_constraint.inverse_transform(value))

    @property
    def mixture_weights(self):
        return self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)

    @mixture_weights.setter
    def mixture_weights(self, value):
        self._set_mixture_weights(value)

    def _set_mixture_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_weights)
        self.initialize(raw_mixture_weights=self.raw_mixture_weights_constraint.inverse_transform(value))

    def initialize_from_data_empspect(self, train_x, train_y):
        """
        Initialize mixture components based on the empirical spectrum of the data.

        This will often be better than the standard initialize_from_data method.
        """
        import numpy as np
        from scipy.fftpack import fft
        from scipy.integrate import cumtrapz

        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")
        if train_x.ndimension() == 1:
            train_x = train_x.unsqueeze(-1)

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

        self.mixture_means = means
        self.mixture_scales = varz
        self.mixture_weights = weights

    def initialize_from_data(self, train_x, train_y, **kwargs):
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")
        if train_x.ndimension() == 1:
            train_x = train_x.unsqueeze(-1)
        if train_x.ndimension() == 2:
            train_x = train_x.unsqueeze(0)

        train_x_sort = train_x.sort(1)[0]
        max_dist = train_x_sort[:, -1, :] - train_x_sort[:, 0, :]
        min_dist_sort = (train_x_sort[:, 1:, :] - train_x_sort[:, :-1, :]).squeeze(0)
        min_dist = torch.zeros(1, self.ard_num_dims, dtype=train_x.dtype, device=train_x.device)
        for ind in range(self.ard_num_dims):
            min_dist[:, ind] = min_dist_sort[((min_dist_sort[:, ind]).nonzero(as_tuple=False))[0], ind]

        # Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2) |
        self.raw_mixture_scales.data.normal_().mul_(max_dist).abs_().pow_(-1)
        self.raw_mixture_scales.data = self.raw_mixture_scales_constraint.inverse_transform(
            self.raw_mixture_scales.data
        )
        # Draw means from Unif(0, 0.5 / minimum distance between two points)
        self.raw_mixture_means.data.uniform_().mul_(0.5).div_(min_dist)
        self.raw_mixture_means.data = self.raw_mixture_means_constraint.inverse_transform(self.raw_mixture_means.data)
        # Mixture weights should be roughly the stdv of the y values divided by the number of mixtures
        self.raw_mixture_weights.data.fill_(train_y.std() / self.num_mixtures)
        self.raw_mixture_weights.data = self.raw_mixture_weights_constraint.inverse_transform(
            self.raw_mixture_weights.data
        )

    def _create_input_grid(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        This is a helper method for creating a grid of the kernel's inputs.
        Use this helper rather than maually creating a meshgrid.

        The grid dimensions depend on the kernel's evaluation mode.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`)
            :attr:`x2` (Tensor `m x d` or `b x m x d`) - for diag mode, these must be the same inputs

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the gridded `x1` and `x2`.
            The shape depends on the kernel's mode

            * `full_covar`: (`b x n x 1 x d` and `b x 1 x m x d`)
            * `full_covar` with `last_dim_is_batch=True`: (`b x k x n x 1 x 1` and `b x k x 1 x m x 1`)
            * `diag`: (`b x n x d` and `b x n x d`)
            * `diag` with `last_dim_is_batch=True`: (`b x k x n x 1` and `b x k x n x 1`)
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

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
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
        x1_exp_, x2_exp_ = self._create_input_grid(
            x1_exp, x2_exp, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )
        x1_cos_, x2_cos_ = self._create_input_grid(
            x1_cos, x2_cos, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )

        # Compute the exponential and cosine terms
        exp_term = (x1_exp_ - x2_exp_).pow_(2).mul_(-2 * math.pi ** 2)
        cos_term = (x1_cos_ - x2_cos_).mul_(2 * math.pi)
        res = exp_term.exp_() * cos_term.cos_()

        # Product over dimensions
        if last_dim_is_batch:
            res = res.squeeze(-1)
        else:
            res = res.prod(-1)

        # Sum over mixtures
        mixture_weights = self.mixture_weights.unsqueeze(-1)
        if not diag:
            mixture_weights = mixture_weights.unsqueeze(-1)
        if last_dim_is_batch:
            mixture_weights = mixture_weights.unsqueeze(-1)

        res = (res * mixture_weights).sum(-2 if diag else -3)
        return res
