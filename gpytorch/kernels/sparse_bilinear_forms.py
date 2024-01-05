#!/usr/bin/env python3

from typing import Callable, Optional, Tuple

import torch
from torch import LongTensor, Tensor

from ..typing import Float, Long


def _scatter_gradients(
    X: Float[Tensor, "... N D"],
    X_grads_unreduced: Float[Tensor, "*batch K I1 I2 D"],
    Si: Long[LongTensor, "*batch K I1 I2"],
):
    batch_shape = X_grads_unreduced.shape[:-4]
    N, D = X.shape[-2:]
    K, I1, I2 = X_grads_unreduced.shape[-4:-1]

    # Prepare X_grads_unreduced and Si for a vmap
    X_grads_unreduced = X_grads_unreduced.reshape(*batch_shape, K * I1 * I2, D)  # *batch x KI^2 x D
    Si = Si.reshape(*batch_shape, -1)  # *batch x KI^2

    # Function to reduce X_grads_unreduced
    def reduce_grad(X_grads_unreduced_sub: Float[Tensor, "... K*I1*I2"]):
        res = torch.zeros(*batch_shape, N, device=X.device)
        res = torch.scatter_reduce(res, -1, Si, X_grads_unreduced_sub, reduce="sum")
        return res

    # Run vmap to reduce X_grads_unreduced
    X_grad = torch.vmap(reduce_grad, in_dims=-1, out_dims=-1)(X_grads_unreduced)
    return X_grad


def _vmap_index_select(
    input: Float[Tensor, "*batch M N"],
    index: Float[Tensor, "*batch K"],
) -> Float[Tensor, "*batch K L"]:
    """ """
    if input.dim() == 2:
        return input[index]
    else:
        return torch.vmap(_vmap_index_select)(input, index)


class SparseBilinearForms(torch.autograd.Function):
    r"""
    An autograd function to compute the bilinear forms
    .. math::

        \boldsymbol S_1^\top \boldsymbol K(\boldsymbol X, \boldsymbol X) \boldsymbol S_2   \quad \text{and} \quad   \boldsymbol S_1^\top \boldsymbol S_2^\top,

    where :math:`\boldsymbol S_1` and :math:`\boldsymbol S_2` are row-wise sparse.
    They are represented by row-wise value/index matrices.

    The kernel matrix :math:`\boldsymbol K(\cdot, \cdot)`
    is given by a forward (and an optional backward) function.

    :param X: The inputs :math:`\boldsymbol X`.
    :param Sv1: The `K1` non-zero values of each row in :math:`\boldsymbol S_1`.
    :param Sv2: The `K2` non-zero values of each row in :math:`\boldsymbol S_2`.
    :param Si1: The indicies of the `K1` non-zero entries of each row in :math:`\boldsymbol S_1`.
    :param Si2: The indicies of the `K2` non-zero entries of each row in :math:`\boldsymbol S_2`.
    :param kernel_forward: The forward function for :math:`\boldsymbol K(\cdot, \cdot)`.
    :param kernel_vjp: A VJP function for :math:`\boldsymbol K(\cdot, \cdot)`.
    :param S2_chunk_size: An optional chunk size (number of columns of :math:`\boldsymbol S_2`
        to compute in parallel) for vmaps.
    """  # noqa: E501

    @staticmethod
    def forward(
        ctx,
        X: Float[Tensor, "... N D"],
        Sv1: Float[Tensor, "... K1 I1"],
        Sv2: Float[Tensor, "... K2 I2"],
        Si1: Long[LongTensor, "... K1 I1"],
        Si2: Long[LongTensor, "... K2 I2"],
        kernel_forward: Callable[[Float[Tensor, "... M D"], Float[Tensor, "... N D"]], Float[Tensor, "... M N"]],
        kernel_vjp: Optional[
            Callable[
                [Float[Tensor, "... M N"], Float[Tensor, "... M D"], Float[Tensor, "... N D"]],
                Tuple[Float[Tensor, "... M D"], Float[Tensor, "... N D"]],
            ]
        ],
        S2_chunk_size: Optional[int] = None,
    ) -> Float[Tensor, "... I1 I2"]:
        """
        O(K^2 I^2) time
        O(max(K^2, I^2)) memory (if vmap is done sequentially)
        """

        # Get consistent batch sizes
        K1, I1 = Sv1.shape[-2:]
        K2, I2 = Sv2.shape[-2:]
        batch_shape = torch.broadcast_shapes(
            X.shape[:-2], Sv1.shape[:-2], Sv2.shape[:-2], Si1.shape[:-2], Si2.shape[:-2]
        )
        X_ = X.expand(*batch_shape, *X.shape[-2:])
        Sv1_ = Sv1[..., :, None].expand(*batch_shape, K1, I1, I2)
        Sv2_ = Sv2[..., None, :].expand(*batch_shape, K2, I1, I2)
        Si1_ = Si1[..., :, None].expand(*batch_shape, K1, I1, I2)
        Si2_ = Si2[..., None, :].expand(*batch_shape, K2, I1, I2)

        # Define a vmap function for forward
        # This function essentially computes s_i^T K s_j and s_i^T s_j
        def _sub_forward(
            Sv1_sub: Float[Tensor, "... K1"],
            Sv2_sub: Float[Tensor, "... K2"],
            Si1_sub: Long[LongTensor, "... K1"],
            Si2_sub: Long[LongTensor, "... K2"],
        ) -> Float[Tensor, "..."]:
            X1_sub = _vmap_index_select(X_, Si1_sub)
            X2_sub = _vmap_index_select(X_, Si2_sub)
            K_sub = kernel_forward(X1_sub, X2_sub)
            Dirac_sub = torch.eq(Si1_sub[..., :, None], Si2_sub[..., None, :]).to(dtype=K_sub.dtype)
            siT_K_sj = (Sv1_sub * (K_sub @ Sv2_sub[..., None]).squeeze(-1)).sum(dim=-1)
            siT_sj = (Sv1_sub * (Dirac_sub @ Sv2_sub[..., None]).squeeze(-1)).sum(dim=-1)
            return siT_K_sj, siT_sj

        # Call s_i^T K s_j and s_i^T s_j for all i, j
        forward = torch.vmap(
            torch.vmap(_sub_forward, in_dims=-1, out_dims=-1), in_dims=-1, out_dims=-1, chunk_size=S2_chunk_size
        )
        S1T_K_S2, S1T_S2 = forward(Sv1_, Sv2_, Si1_, Si2_)

        # Maybe save stuff for the backward pass.
        ctx.save_for_backward(X, Sv1, Sv2, Si1, Si2)
        ctx.S2_chunk_size = S2_chunk_size
        if ctx.needs_input_grad[0]:
            if kernel_vjp is None:
                kernel_vjp = lambda V, X1, X2: torch.autograd.functional.vjp(
                    lambda X1, X2: kernel_forward(X1, X2), (X1, X2), V
                )[1]
            ctx.kernel_vjp = kernel_vjp
        if any(ctx.needs_input_grad[1:3]):
            ctx.forward = forward
            ctx.kernel_forward = kernel_forward

        # Done!
        return S1T_K_S2, S1T_S2

    @staticmethod
    def backward(ctx, V_S1T_K_S2, V_S1T_S2):
        """
        O(K^2 I^2) time
        O(I^2 K D) memory
        """
        # Get consistent batch sizes
        X, Sv1, Sv2, Si1, Si2 = ctx.saved_tensors
        K1, I1 = Sv1.shape[-2:]
        K2, I2 = Sv2.shape[-2:]
        batch_shape = torch.broadcast_shapes(
            X.shape[:-2], Sv1.shape[:-2], Sv2.shape[:-2], Si1.shape[:-2], Si2.shape[:-2]
        )
        X_ = X.expand(*batch_shape, *X.shape[-2:])
        Sv1_ = Sv1[..., :, None].expand(*batch_shape, K1, I1, I2)
        Sv2_ = Sv2[..., None, :].expand(*batch_shape, K2, I1, I2)
        Si1_ = Si1[..., :, None].expand(*batch_shape, K1, I1, I2)
        Si2_ = Si2[..., None, :].expand(*batch_shape, K2, I1, I2)

        # X
        if ctx.needs_input_grad[0]:
            # Define a vmap function for backward
            # This function essentially computes d(v_ij * s_i^T K s_j) / d(X_1), and likewise for X_2
            def _sub_backward(
                Sv1_sub: Float[Tensor, "... K1"],
                Sv2_sub: Float[Tensor, "... K2"],
                Si1_sub: Long[LongTensor, "... K1"],
                Si2_sub: Long[LongTensor, "... K2"],
            ) -> Float[Tensor, "... K1+K2 D"]:
                X1_sub = _vmap_index_select(X_, Si1_sub)
                X2_sub = _vmap_index_select(X_, Si2_sub)
                return ctx.kernel_vjp((Sv1_sub[..., :, None] @ Sv2_sub[..., None, :]), X1_sub, X2_sub)

            # Call d(v_ij * s_i^T K s_j) / dX for all i, j, and sum
            backward = torch.vmap(
                torch.vmap(_sub_backward, in_dims=-1, out_dims=-2),
                in_dims=-1,
                out_dims=-2,
                chunk_size=ctx.S2_chunk_size,
            )
            X1_grads, X2_grads = backward(Sv1_, Sv2_, Si1_, Si2_)  # ... x k*2 x i x i x d
            X_grad = torch.add(
                _scatter_gradients(X, V_S1T_K_S2[..., None, :, :, None] * X1_grads, Si1_),
                _scatter_gradients(X, V_S1T_K_S2[..., None, :, :, None] * X2_grads, Si2_),
            )
        else:
            X_grad = None

        # Sv1
        if ctx.needs_input_grad[1]:
            # d tr(V^T (S1^T K S2) / d S1
            # = d tr(S1^T K S2 V^T) / d S1
            # = K S2 V^T
            # Similarly, d tr(V^T S1^T S2) / d S1 = S2 V^T
            # We can use the "forward" helper to compute `I K S2` and `I S2`
            Sv1_full_grad_a, Sv1_full_grad_b = ctx.forward(
                torch.ones_like(Sv1_).unsqueeze(-3),
                Sv2_.unsqueeze(-4),
                Si1_.unsqueeze(-3),
                Si2_.unsqueeze(-4),
            )
            Sv1_grad = torch.addcmul(
                torch.mul(Sv1_full_grad_a, V_S1T_K_S2.unsqueeze(-3)),
                Sv1_full_grad_b,
                V_S1T_S2.unsqueeze(-3),
            ).sum(dim=-1)
        else:
            Sv1_grad = None

        # Sv2
        if ctx.needs_input_grad[2]:
            # d tr(V^T (S1^T K S2) / d S2
            # = d tr(S2^T K^T S1 V) / d S1
            # = K^T S1 V
            # Similarly, d tr(V^T S1^T S2) / d S2 = S1 V
            # We can use the "forward" helper to compute `I K^T S1` and `I S1`
            Sv2_full_grad_a, Sv2_full_grad_b = ctx.forward(
                Sv1_.unsqueeze(-4),
                torch.ones_like(Sv2_).unsqueeze(-3),
                Si1_.unsqueeze(-4),
                Si2_.unsqueeze(-3),
            )
            Sv2_grad = torch.addcmul(
                torch.mul(Sv2_full_grad_a, V_S1T_K_S2.unsqueeze(-3)),
                Sv2_full_grad_b,
                V_S1T_S2.unsqueeze(-3),
            ).sum(dim=-2)
        else:
            Sv2_grad = None

        return (
            X_grad,
            Sv1_grad,
            Sv2_grad,
            None,  # Si1
            None,  # Si2
            None,  # kernel_forward
            None,  # kernel_vjp
            None,  # S2_chunk_size
        )
