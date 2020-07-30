import math

import torch

from ..constraints import Positive
from ..lazy import MatmulLazyTensor, RootLazyTensor
from .kernel import Kernel


class SpectralDeltaKernel(Kernel):
    """
    A kernel that supports spectral learning for GPs, where the underlying spectral density is modeled as a mixture
    of delta distributions (e.g., with point masses). This has been explored e.g. in Lazaro-Gredilla et al., 2010.

    Conceptually, this kernel is similar to random Fourier features as implemented in RFFKernel, but instead of sampling
    a Gaussian to determine the spectrum sites, they are treated as learnable parameters.

    When using CG for inference, this kernel supports linear space and time (in N) for training and inference.

    :param int num_dims: Dimensionality of input data that this kernel will operate on. Note that if active_dims is
        used, this should be the length of the active dim set.
    :param int num_deltas: Number of point masses to learn.
    """

    has_lengthscale = True

    def __init__(self, num_dims, num_deltas=128, Z_constraint=None, batch_shape=torch.Size([]), **kwargs):
        Kernel.__init__(self, has_lengthscale=True, batch_shape=batch_shape, **kwargs)

        self.raw_Z = torch.nn.Parameter(torch.rand(*batch_shape, num_deltas, num_dims))

        if Z_constraint:
            self.register_constraint("raw_Z", Z_constraint)
        else:
            self.register_constraint("raw_Z", Positive())

        self.num_dims = num_dims

    def initialize_from_data(self, train_x, train_y):
        """
        Initialize the point masses for this kernel from the empirical spectrum of the data. To do this, we estimate
        the empirical spectrum's CDF and then simply sample from it. This is analogous to how the SM kernel's mixture
        is initialized, but we skip the last step of fitting a GMM to the samples and just use the samples directly.
        """
        import numpy as np
        from scipy.fftpack import fft
        from scipy.integrate import cumtrapz

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

        a = np.random.rand(self.raw_Z.size(-2), 1)
        p, q = np.histogram(a, spec_cdf)
        bins = np.digitize(a, q)
        slopes = (spec_cdf[bins] - spec_cdf[bins - 1]) / (freq[bins] - freq[bins - 1])
        intercepts = spec_cdf[bins - 1] - slopes * freq[bins - 1]
        inv_spec = (a - intercepts) / slopes

        self.Z = inv_spec

    def initialize_from_data_simple(self, train_x, train_y, **kwargs):
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")
        if train_x.ndimension() == 1:
            train_x = train_x.unsqueeze(-1)
        if train_x.ndimension() == 2:
            train_x = train_x.unsqueeze(0)

        train_x_sort = train_x.sort(1)[0]
        min_dist_sort = (train_x_sort[:, 1:, :] - train_x_sort[:, :-1, :]).squeeze(0)
        ard_num_dims = 1 if self.ard_num_dims is None else self.ard_num_dims
        min_dist = torch.zeros(1, ard_num_dims, dtype=self.Z.dtype, device=self.Z.device)
        for ind in range(ard_num_dims):
            min_dist[:, ind] = min_dist_sort[(torch.nonzero(min_dist_sort[:, ind]))[0], ind]

        z_init = torch.rand_like(self.Z).mul_(0.5).div_(min_dist)

        self.Z = z_init

    @property
    def Z(self):
        return self.raw_Z_constraint.transform(self.raw_Z)

    @Z.setter
    def Z(self, value):
        self._set_Z(value)

    def _set_Z(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_Z)
        self.initialize(raw_Z=self.raw_Z_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        Z = self.Z

        # Z1_ and Z2_ are s x d
        x1z1 = x1_.matmul(Z.transpose(-2, -1))  # n x s
        x2z2 = x2_.matmul(Z.transpose(-2, -1))  # n x s

        x1z1 = x1z1 * 2 * math.pi
        x2z2 = x2z2 * 2 * math.pi

        x1z1 = torch.cat([x1z1.cos(), x1z1.sin()], dim=-1) / math.sqrt(x1z1.size(-1))
        x2z2 = torch.cat([x2z2.cos(), x2z2.sin()], dim=-1) / math.sqrt(x2z2.size(-1))

        if x1.size() == x2.size() and torch.equal(x1, x2):
            prod = RootLazyTensor(x1z1)
        else:
            prod = MatmulLazyTensor(x1z1, x2z2.transpose(-2, -1))

        if diag:
            return prod.diag()
        else:
            return prod
