from typing import Optional, Union

import torch

from jaxtyping import Float
from linear_operator import LinearOperator, to_dense
from torch import Tensor


def sum_interaction_terms(
    covars: Float[Union[LinearOperator, Tensor], "... D N N"],
    max_degree: Optional[int] = None,
    dim: int = -3,
) -> Float[Tensor, "... N N"]:
    r"""
    Given a batch of D x N x N covariance matrices :math:`\boldsymbol K_1, \ldots, \boldsymbol K_D`,
    compute the sum of each covariance matrix as well as the interaction terms up to degree `max_degree`
    (denoted as :math:`M` below):

    .. math::

        \sum_{1 \leq i_1 < i_2 < \ldots <  i_M < D} \left[
            \prod_{j=1}^M \boldsymbol K_{i_j}
        \right].

    This function is useful for computing the sum of additive kernels as defined in
    `Additive Gaussian Processes (Duvenaud et al., 2011)`_.

    Note that the summation is computed in :math:`\mathcal O(D)` time using the Newton-Girard formula.

    .. _Additive Gaussian Processes (Duvenaud et al., 2011):
        https://arxiv.org/pdf/1112.4394

    :param covars: A batch of covariance matrices, representing the base covariances to sum over
    :param max_degree: The maximum degree of the interaction terms to compute.
        If not provided, this will default to `D`.
    :param dim: The dimension to sum over (i.e. the batch dimension containing the base covariance matrices).
        Note that dim must be a negative integer (i.e. -3, not 0).
    """
    if dim >= 0:
        raise ValueError("Argument 'dim' must be a negative integer.")

    covars = to_dense(covars)
    ks = torch.arange(max_degree, dtype=covars.dtype, device=covars.device)
    neg_one = torch.tensor(-1.0, dtype=covars.dtype, device=covars.device)

    # S_times_factor[k] = factor[k] * S[k]
    #                   = (-1)^{k} * \sum_{i=1}^D covar_i^{k+1}
    S_times_factor_ks = torch.vmap(lambda k: neg_one.pow(k) * torch.sum(covars.pow(k + 1), dim=dim))(ks)

    # E[deg] = 1/(deg+1) \sum_{j=0}^{deg} factor[k] * S[k] * E[deg-k]
    #           = 1/(deg+1) [ (factor[deg] * S[deg]) + \sum_{j=1}^{deg - 1} factor * S_ks[k] * E_ks[deg-k] ]
    E_ks = torch.empty_like(S_times_factor_ks)
    E_ks[0] = S_times_factor_ks[0]
    for deg in range(1, max_degree):
        sum_term = torch.einsum("m...,m...->...", S_times_factor_ks[:deg], E_ks[:deg].flip(0))
        E_ks[deg] = (S_times_factor_ks[deg] + sum_term) / (deg + 1)

    return E_ks.sum(0)
