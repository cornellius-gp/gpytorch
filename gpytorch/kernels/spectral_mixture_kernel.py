from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import math
import torch
from .kernel import Kernel

logger = logging.getLogger()


class SpectralMixtureKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Spectral Mixture Kernel
    between inputs :math:`mathbf{x_1}` and :math:`mathbf{x_2}`:
    It was proposed in `Gaussian Process Kernels for Pattern Discovery and Extrapolation`_.

    .. note::

        Unlike other kernels,
        * :attr:`ard_num_dums` **must equal** the number of dimensions of the data
        * :attr:`batch_size` **must equal** the batch size of the data (1 if the data is not batched)
        * This kernel should not be combined with a :class:`gpytorch.kernels.ScaleKernel`.

    Args:
        :attr:`num_mixtures` (int, optional):
            The number of components in the mixture.
        :attr:`ard_num_dims` (int, optional):
            Set this to match the dimensionality of the input.
            It should be `d` if :attr:`x1` is a `n x d` matrix. Default: `1`
        :attr:`batch_size` (int, optional):
            Set this if the data is
            batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `1`
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to
            compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take
            (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`mixture_lengthscale` (Tensor):
            The lengthscale parameter. Given `k` mixture components,
            and `b x n x d` data, this will be of size `b x k x 1 x d`.
        :attr:`mixture_means` (Tensor):
            The mixture mean parameters (`b x k x 1 x d`).
        :attr:`mixture_weights` (Tensor):
            The mixture weight parameters (`b x k`).

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=5))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_size=2))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)


    .. _Gaussian Process Kernels for Pattern Discovery and Extrapolation:
        https://arxiv.org/pdf/1302.4245.pdf
    """
    # TODO: add equation to docs

    def __init__(
        self,
        num_mixtures=None,
        ard_num_dims=1,
        batch_size=1,
        active_dims=None,
        eps=1e-6,
        log_mixture_scales_prior=None,
        log_mixture_means_prior=None,
        log_mixture_weights_prior=None,
    ):
        if num_mixtures is None:
            raise RuntimeError("num_mixtures is a required argument")
        if (
            log_mixture_means_prior is not None
            or log_mixture_scales_prior is not None
            or log_mixture_weights_prior is not None
        ):
            logger.warning("Priors not implemented for SpectralMixtureKernel")

        # This kernel does not use the default lengthscale
        super(SpectralMixtureKernel, self).__init__(active_dims=active_dims)
        self.num_mixtures = num_mixtures
        self.batch_size = batch_size
        self.ard_num_dims = ard_num_dims
        self.eps = eps

        self.register_parameter(
            name="log_mixture_weights", parameter=torch.nn.Parameter(torch.zeros(self.batch_size, self.num_mixtures))
        )
        self.register_parameter(
            name="log_mixture_means",
            parameter=torch.nn.Parameter(torch.zeros(self.batch_size, self.num_mixtures, 1, self.ard_num_dims)),
        )
        self.register_parameter(
            name="log_mixture_scales",
            parameter=torch.nn.Parameter(torch.zeros(self.batch_size, self.num_mixtures, 1, self.ard_num_dims)),
        )

    @property
    def mixture_scales(self):
        return self.log_mixture_scales.exp().clamp(self.eps, 1e5)

    @property
    def mixture_means(self):
        return self.log_mixture_means.exp().clamp(self.eps, 1e5)

    @property
    def mixture_weights(self):
        return self.log_mixture_weights.exp().clamp(self.eps, 1e5)

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
        min_dist = torch.zeros(1, self.ard_num_dims)
        for ind in range(self.ard_num_dims):
            min_dist[:, ind] = min_dist_sort[(torch.nonzero(min_dist_sort[:, ind]))[0], ind]

        # Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2) |
        self.log_mixture_scales.data.normal_().mul_(max_dist).abs_().pow_(-1).log_()
        # Draw means from Unif(0, 0.5 / minimum distance between two points)
        self.log_mixture_means.data.uniform_().mul_(0.5).div_(min_dist).log_()
        # Mixture weights should be roughly the stdv of the y values divided by the number of mixtures
        self.log_mixture_weights.data.fill_(train_y.std() / self.num_mixtures).log_()

    def forward(self, x1, x2, **params):
        batch_size, n, num_dims = x1.size()
        _, m, _ = x2.size()

        if not num_dims == self.ard_num_dims:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have {} dimensionality "
                "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, num_dims)
            )
        if not batch_size == self.batch_size:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have a batch_size of {} "
                "(based on the batch_size argument). Got {}.".format(self.batch_size, batch_size)
            )

        # Expand x1 and x2 to account for the number of mixtures
        # Should make x1/x2 (b x k x n x d) for k mixtures
        x1_ = x1.unsqueeze(1)
        x2_ = x2.unsqueeze(1)

        # Compute distances - scaled by appropriate parameters
        x1_exp = x1_ * self.mixture_scales
        x2_exp = x2_ * self.mixture_scales
        x1_cos = x1_ * self.mixture_means
        x2_cos = x2_ * self.mixture_means

        # Create grids
        x1_exp_, x2_exp_ = self._create_input_grid(x1_exp, x2_exp, **params)
        x1_cos_, x2_cos_ = self._create_input_grid(x1_cos, x2_cos, **params)

        # Compute the exponential and cosine terms
        exp_term = (x1_exp_ - x2_exp_).pow_(2).mul_(-2 * math.pi ** 2)
        cos_term = (x1_cos_ - x2_cos_).mul_(2 * math.pi)
        res = exp_term.exp_() * cos_term.cos_()

        # Product omer dimensions
        res = res.prod(-1)

        # Sum over mixtures
        mixture_weights = self.mixture_weights
        while mixture_weights.dim() < res.dim():
            mixture_weights.unsqueeze_(-1)
        res = (res * mixture_weights).sum(1)
        return res
