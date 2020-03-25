import math

import numpy as np
import torch
from scipy.special import ellipj, ellipk

from .. import settings
from .lanczos import lanczos_tridiag
from .minres import minres


def contour_integral_quad(
    lazy_tensor, rhs, inverse=False, weights=None, shifts=None, max_lanczos_iter=10, num_contour_quadrature=7
):
    r"""
    Performs :math:`\mathbf K^{1/2} \mathbf b` or `\mathbf K^{-1/2} \mathbf b`
    using contour integral quadrature.

    .. note::
        Right now this only works for non-batch matrices

    :param gpytorch.lazy.LazyTensor lazy_tensor: LazyTensor representing :math:`\mathbf K`
    :param torch.Tensor rhs: Right hand side tensor :math:`\mathbf b`
    :param bool inverse: (default False) whether to compute :math:`\mathbf K^{1/2} \mathbf b` (if False)
        or `\mathbf K^{-1/2} \mathbf b` (if True)
    :param int max_lanczos_iter: (default 10) Number of Lanczos iterations to run (to estimate eigenvalues)
    :param int num_contour_quadrature: (default 15) How many quadrature samples to use for approximation
    :rtype: torch.Tensor
    :return: Approximation to :math:`\mathbf K^{1/2} \mathbf b` or :mathbf:`\mathbf K^{-1/2} \mathbf b`.
    """
    if len(lazy_tensor.batch_shape):
        raise RuntimeError("CIQ/Sqrt Inv Matmul only works for non-batch matrices ATM.")

    if shifts is None:
        rhs_index = tuple([0] * (rhs.dim() - 2) + [slice(None, None, None), slice(None, 1, None)])
        lanczos_basis, lanczos_mat = lanczos_tridiag(
            lambda v: lazy_tensor._matmul(v),
            init_vecs=rhs[rhs_index],
            dtype=rhs.dtype,
            device=rhs.device,
            matrix_shape=lazy_tensor.matrix_shape,
            batch_shape=lazy_tensor.batch_shape,
            max_iter=max_lanczos_iter,
        )

        """
        K^{-1/2} b = 2/pi \int_0^\infty (K - t^2 I)^{-1} dt
        We'll approximate this integral as a sum using quadrature
        We'll determine the appropriate values of t, as well as their weights using elliptical integrals
        """

        # Compute an approximate condition number
        # We'll do this with Lanczos
        approx_eigs = lanczos_mat.symeig()[0]
        approx_eigs = approx_eigs[approx_eigs > 0]
        min_eig = approx_eigs.min()
        max_eig = approx_eigs.max()
        k2 = min_eig / max_eig
        if settings.record_ciq_stats.on():
            settings.record_ciq_stats.condition_number = 1.0 / k2

        # Compute the shifts needed for the contour
        Kp = ellipk(1 - k2.detach().cpu().numpy())  # Elliptical integral of the first kind
        N = num_contour_quadrature
        t = 1j * (np.arange(1, N + 1) - 0.5) * Kp / N
        sn, cn, dn, _ = ellipj(np.imag(t), 1 - k2.item())  # Jacobi elliptic functions
        cn = 1.0 / cn
        dn = dn * cn
        sn = 1j * sn * cn
        w = np.sqrt(min_eig.item()) * sn
        w_pow2 = np.real(np.power(w, 2))
        shifts = torch.from_numpy(w_pow2.astype(float)).type_as(rhs).to(rhs.device)

    # Compute the solves at the given shifts
    # Do one more matmul if we don't want to include the inverse
    extra_shifts = torch.cat([torch.tensor([0.0], dtype=shifts.dtype, device=shifts.device), shifts])
    solves = minres(lambda v: lazy_tensor._matmul(v), rhs, value=-1, shifts=extra_shifts)
    no_shift_solves = solves[0]
    solves = solves[1:]
    if not inverse:
        solves = lazy_tensor._matmul(solves)

    # Compute the weights that correspond to the different
    # These weights include the constant 2/pi
    if weights is None:
        constant = -2 * Kp * np.sqrt(min_eig.item()) / (math.pi * N)
        dzdt = cn * dn
        dzdt_th = torch.from_numpy(dzdt).type_as(solves)
        dzdt_th = dzdt_th.view(dzdt_th.numel(), *([1] * (solves.dim() - 1)))
        dzdt_th.mul_(constant)
        weights = dzdt_th

    # Record some stats on how good the solves are
    if settings.record_ciq_stats.on():
        with torch.no_grad():
            settings.record_ciq_stats.minres_residual = (
                (lazy_tensor @ no_shift_solves + rhs)
                .div_(rhs.norm(dim=-2, keepdim=True).clamp_min_(1e-10))
                .norm(dim=-2)
                .mean()
                .item()
            )

    return solves, weights, no_shift_solves, shifts
