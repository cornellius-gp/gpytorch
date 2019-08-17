from scipy.special import ellipk, ellipj
from .lanczos import lanczos_tridiag
from .minres import minres
import torch
import numpy as np
import math


def sqrt_matmul(lazy_tensor, rhs, inverse=False, max_lanczos_iter=10, num_quad_samples=15):
    """
    Performs A^{1/2} rhs using contour integral quadrature.

    Args:
        - lazy_tensor (gpytorch.lazy.LazyTensor): LazyTensor representing A
        - rhs (torch.Tensor): rhs to multiply with
        - min_eig_est (torch.Tensor): Estimated minimum eigenvalue

    Returns:
        - A very good approximation to A^{1/2} rhs
    """
    lanczos_basis, lanczos_mat = lanczos_tridiag(
        lambda v: lazy_tensor._matmul(v),
        init_vecs=rhs,
        dtype=rhs.dtype,
        device=rhs.device,
        matrix_shape=lazy_tensor.shape,
        max_iter=5,
    )

    # We need to run Lanczos anyways
    if lanczos_mat.dim() > 2:
        ind = [0] * (lanczos_mat.dim() - 2)
        approx_eigs = lanczos_mat.__getitem__(*ind).symeig()[0]
    else:
        approx_eigs = lanczos_mat.symeig()[0]
    approx_eigs = approx_eigs[approx_eigs > 0]
    min_eig = approx_eigs.min()
    max_eig = approx_eigs.max()

    k2 = min_eig / max_eig

    Kp = ellipk(1 - k2.detach().cpu().numpy())  # Elliptical integral of the first kind
    N = 15
    t = 1j * (np.arange(1, N + 1) - 0.5) * Kp / N
    sn, cn, dn, _ = ellipj(np.imag(t), 1 - k2.item())  # Jacobi elliptic functions
    cn = 1. / cn
    dn = dn * cn
    sn = 1j * sn * cn
    w = np.sqrt(min_eig.item()) * sn
    dzdt = cn * dn
    w_pow2 = np.real(np.power(w, 2))
    solves = minres(
        lambda v: lazy_tensor._matmul(v),
        rhs,
        value=-1,
        shifts=torch.from_numpy(w_pow2.astype(float)).type_as(rhs).to(rhs.device),
    )

    dzdt_th = torch.from_numpy(dzdt).type_as(solves)
    dzdt_th = dzdt_th.view(dzdt_th.numel(), *([1] * (solves.dim() - 1)))
    weighted_solves = dzdt_th * solves
    summed_solves = weighted_solves.sum(0)

    if not inverse:
        res = lazy_tensor._matmul(summed_solves)
    else:
        res = summed_solves

    constant = -2 * Kp * np.sqrt(min_eig.item()) / (math.pi * N)
    res = constant * res
    return res.squeeze(-1)
