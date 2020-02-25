import math

import numpy as np
import torch
from scipy.special import ellipj, ellipk

from .lanczos import lanczos_tridiag
from .minres import minres


def contour_integral_quad(lazy_tensor, rhs, inverse=False, max_lanczos_iter=5, num_quad_samples=7):
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
    :param int num_quad_samples: (default 15) How many quadrature samples to use for approximation
    :rtype: torch.Tensor
    :return: Approximation to :math:`\mathbf K^{1/2} \mathbf b` or :mathbf:`\mathbf K^{-1/2} \mathbf b`.
    """
    if len(lazy_tensor.batch_shape):
        raise RuntimeError("CIQ/Sqrt Inv Matmul only works for non-batch matrices ATM.")

    lanczos_basis, lanczos_mat = lanczos_tridiag(
        lambda v: lazy_tensor._matmul(v),
        init_vecs=rhs,
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
    if lanczos_mat.dim() > 2:
        ind = [0] * (lanczos_mat.dim() - 2)
        approx_eigs = lanczos_mat.__getitem__(ind).symeig()[0]
    else:
        approx_eigs = lanczos_mat.symeig()[0]
    approx_eigs = approx_eigs[approx_eigs > 0]
    min_eig = approx_eigs.min()
    max_eig = approx_eigs.max()
    k2 = min_eig / max_eig

    # Compute the shifts needed for the contour
    Kp = ellipk(1 - k2.detach().cpu().numpy())  # Elliptical integral of the first kind
    N = 15
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
    extra_shifts = torch.cat([torch.zeros(1, dtype=shifts.dtype, device=shifts.device), shifts])
    solves = minres(lambda v: lazy_tensor._matmul(v), rhs, value=-1, shifts=extra_shifts, max_iter=100)
    if not inverse:
        solves = lazy_tensor._matmul(solves)
    no_shift_solves = solves[0]
    solves = solves[1:]

    # Compute the weights that correspond to the different
    # These weights include the constant 2/pi
    constant = -2 * Kp * np.sqrt(min_eig.item()) / (math.pi * N)
    dzdt = cn * dn
    dzdt_th = torch.from_numpy(dzdt).type_as(solves)
    dzdt_th = dzdt_th.view(dzdt_th.numel(), *([1] * (solves.dim() - 1)))
    dzdt_th.mul_(constant)

    return solves, dzdt_th, no_shift_solves
