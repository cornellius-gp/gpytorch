from typing import Optional

import torch
from torch import Tensor


def sparse_chol_inv(pd_mat, nn_k: int, nn_ind: Optional[Tensor] = None):
    """
    Computes a sparse approximation of L^{-1}, where L = cholesky(pd_mat).

    If we want a proper Cholesky factor of the precision matrix pd_mat^{-1}, one can do:

        >>> L_prec = sparse_chol_inv(pd_mat, nn_k).flip(-2).flip(-1).transpose(-2, -1)  # approximates chol(inverse(K))

    .. note::
        We use the ordering 0,...,n so pd_mat should already be permuted to some desired ordering before using this
        function.

    Args:
        pd_mat (torch.Tensor or gpytorch.lazy.LazyTensor): Positive definite input matrix
        nn_k (int): Number of nearest neighbors to use for approximation.
        nn_ind (torch.LongTensor, optional): An n x 3
    """
    from ..lazy import lazify, delazify, DiagLazyTensor

    batch_shape = pd_mat.shape[:-2]

    if batch_shape != torch.Size([]):
        raise RuntimeError(
            "sparse_chol_inv does not currently support batches due to some advanced indexing problems (TODO)."
        )

    # Getting nn_ind here is the only line of code that couldn't execute on arbitrary LazyTensors,
    # since we don't/can't support tril / argsort. If you pass in the equivalent of nn_ind
    if nn_ind is None:
        nn_ind = delazify(pd_mat).tril().argsort(-1, descending=True)[..., :, 1 : (nn_k + 1)]

    pd_mat = lazify(pd_mat)

    # perfect shuffle
    pi = torch.arange(nn_k * nn_k).view(nn_k, nn_k).transpose(-2, -1).contiguous().view(-1)

    # if nn_ind is [0, 2, 4] then left is [0, 2, 4, 0, 2, 4, 0, 2, 4]
    left = nn_ind.repeat(*([1] * len(batch_shape)), 1, nn_k)
    # Apply perfect shuffle so that right is [0, 0, 0, 2, 2, 2, 4, 4, 4]
    right = left[..., :, pi]

    # Indexing with left and right gets us the submatrix of pd_mat corresponding to rows and columns specified by nn_ind
    # TODO: This line below is broken in batch mode.
    pd_mat_nn_blocks = pd_mat[..., left, right]  # n x k^2
    pd_mat_nn_blocks = pd_mat_nn_blocks.view(*pd_mat.shape[:-1], nn_k, nn_k)  # n x k x k

    # pd_mat_nn_blocks is now the n x k x k matrix C(N(x_i), N(x_i))

    # Get the n x k x 1 matrix C(N(x_i), x_i)
    pd_mat_nn_self = pd_mat[..., nn_ind, torch.arange(pd_mat.size(-1), device=pd_mat.device).unsqueeze(-1)].unsqueeze(
        -1
    )

    pd_mat_self_self = pd_mat.diag()

    # Zero out elements corresponding to edge cases (e.g., for i < nn_k)
    pd_mat_nn_self = pd_mat_nn_self.squeeze(-1).tril(-1).unsqueeze(-1)

    # TODO: Might be able to vectorize this for loop?
    # For the first nn_k matrices, we need them to be like [C 0 ; 0 I], where C is the top i x i block
    # of pd_mat_nn_blocks[i]
    for i in range(nn_k):
        pd_mat_nn_blocks[i, i:, :i] = 0.0
        pd_mat_nn_blocks[i, :i, i:] = 0.0
        pd_mat_nn_blocks[i, i:, i:] = torch.eye(nn_k - i, dtype=pd_mat_nn_blocks.dtype, device=pd_mat_nn_blocks.device)

    L_nn_nn = delazify(lazify(pd_mat_nn_blocks).add_jitter(1e-5).cholesky())
    b_vecs = torch.cholesky_solve(pd_mat_nn_self, L_nn_nn)

    f_diag = pd_mat_self_self - (pd_mat_nn_self.transpose(-2, -1) @ b_vecs).squeeze(-1).squeeze(-1)

    B = torch.zeros(*batch_shape, pd_mat.size(-2), pd_mat.size(-1))
    B.scatter_(-1, nn_ind, b_vecs.squeeze(-1)).tril_(-1)

    F_sqrt_inv = DiagLazyTensor(f_diag.sqrt().reciprocal())
    I_minus_B = lazify(-B).add_jitter(1.0).evaluate()

    chol_precision_mat = F_sqrt_inv @ I_minus_B

    return chol_precision_mat
