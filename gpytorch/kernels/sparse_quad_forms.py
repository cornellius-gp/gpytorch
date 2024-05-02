#!/usr/bin/env python3

from typing import Callable, Optional, Tuple

import torch
from torch import LongTensor, Tensor

from ..typing import Float, Long


def _scatter_gradients(
    X: Float[Tensor, "... N D"],
    X_grads_unreduced: Float[Tensor, "*batch K D I I"],
    Si: Long[LongTensor, "*batch K I I"],
):
    batch_shape = X_grads_unreduced.shape[:-4]
    N, D = X.shape[-2:]
    K, _, I = X_grads_unreduced.shape[-4:-1]

    # Prepare X_grads_unreduced and Si for a vmap
    X_grads_unreduced = X_grads_unreduced.transpose(-3, -4).reshape(*batch_shape, D, K * I * I)  # *batch * D x KI^2
    Si = Si.reshape(*batch_shape, -1)  # *batch x KI^2

    # Function to reduce X_grads_unreduced
    def reduce_grad(X_grads_unreduced_sub: Float[Tensor, "... K*I1*I2"]):
        res = torch.zeros(*batch_shape, N, device=X.device)
        res = torch.scatter_reduce(res, -1, Si, X_grads_unreduced_sub, reduce="sum")
        return res

    # Run vmap to reduce X_grads_unreduced
    X_grad = torch.vmap(reduce_grad, in_dims=-2, out_dims=-1)(X_grads_unreduced)
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


class SparseQuadForm(torch.autograd.Function):
    r"""
    An autograd function to compute the bilinear forms
    .. math::

        \boldsymbol S^\top \boldsymbol K(\boldsymbol X, \boldsymbol X) \boldsymbol S

    where :math:`\boldsymbol S` is a row-wise sparse matrix
    that is stored as row-wise value/index matrices.

    The kernel matrix :math:`\boldsymbol K(\cdot, \cdot)`
    is given by a forward (and an optional backward) function.

    :param X: The inputs :math:`\boldsymbol X`.
    :param Sv: The `K` non-zero values of each row in :math:`\boldsymbol S`.
    :param Si: The indicies of the `K` non-zero entries of each row in :math:`\boldsymbol S`.
    :param kernel_forward: The forward function for :math:`\boldsymbol K(\cdot, \cdot)`.
    :param kernel_forward_and_vjp: A function to compute :math:`\boldsymbol K(\cdot, \cdot)` and its VJP.
    :param chunk_size: An optional chunk size (number of columns of
        :math:`\boldsymbol S` to compute in parallel) for vmaps.
    """  # noqa: E501

    @staticmethod
    def forward(
        ctx,
        X: Float[Tensor, "... N D"],
        Sv: Float[Tensor, "... K I"],
        Si: Optional[Long[LongTensor, "... K I"]],
        kernel_forward: Callable[[Float[Tensor, "... M D"], Float[Tensor, "... N D"]], Float[Tensor, "... M N"]],
        kernel_forward_and_vjp: Optional[
            Callable[
                [Float[Tensor, "... M N"], Float[Tensor, "... M D"], Float[Tensor, "... N D"]],
                Tuple[Float[Tensor, "... M D"], Float[Tensor, "... N D"]],
            ]
        ],
        chunk_size: Optional[int] = None,
    ) -> Float[Tensor, "... I I"]:
        """
        O(K^2 I^2) time
        O(max(K^2, I^2)) memory (if vmap is done sequentially)
        """

        # Get consistent batch sizes
        K, I = Sv.shape[-2:]
        N, D = X.shape[-2:]
        batch_shape = torch.broadcast_shapes(X.shape[:-2], Sv.shape[:-2])
        X_ = X.expand(*batch_shape, N, D)
        Sv_ = Sv[..., :, None].expand(*batch_shape, K, I, I)
        if Si is None:
            assert K * I == N
        else:
            Si_ = Si[..., :, None].expand(*batch_shape, K, I, I)

        # Define a vmap function for forward
        # This function essentially computes s_i^T K s_j
        def _sub_forward(
            X1_sub: Float[Tensor, "... K D"],
            X2_sub: Float[Tensor, "... K D"],
            Sv1_sub: Float[Tensor, "... K"],
            Sv2_sub: Float[Tensor, "... K"],
        ) -> Float[Tensor, "..."]:
            K_sub = kernel_forward(X1_sub, X2_sub)
            siT_K_sj = (Sv1_sub * (K_sub @ Sv2_sub[..., None]).squeeze(-1)).sum(dim=-1)
            return siT_K_sj

        def _sub_forward_with_index(
            Sv1_sub: Float[Tensor, "... K"],
            Sv2_sub: Float[Tensor, "... K"],
            Si1_sub: Long[LongTensor, "... K"],
            Si2_sub: Long[LongTensor, "... K"],
        ) -> Float[Tensor, "..."]:
            X1_sub = _vmap_index_select(X_, Si1_sub)
            X2_sub = _vmap_index_select(X_, Si2_sub)
            return _sub_forward(X1_sub, X2_sub, Sv1_sub, Sv2_sub)

        # Call s_i^T K s_j for all i, j
        if Si is None:
            sub_forward_fn = _sub_forward
            X_ = X_.view(*batch_shape, I, K, D).transpose(-3, -2).transpose(-2, -1)  # ... x K x D x I
            X_ = X_[..., None].expand(
                *batch_shape, K, D, I, I
            )  # ... x K x D x I x I, where last every column is identifical
            forward_args = X_, X_.mT, Sv_, Sv_.mT
        else:
            sub_forward_fn = _sub_forward_with_index
            forward_args = Sv_, Sv_.mT, Si_, Si_.mT
        forward = torch.vmap(
            torch.vmap(sub_forward_fn, in_dims=-1, out_dims=-1), in_dims=-1, out_dims=-1, chunk_size=chunk_size
        )
        ST_K_S = forward(*forward_args)

        # Maybe save stuff for the backward pass.
        if any(ctx.needs_input_grad[:]):
            ctx.save_for_backward(X, Sv, Si)
            ctx.chunk_size = chunk_size
        if ctx.needs_input_grad[0]:
            if kernel_forward_and_vjp is None:
                kernel_forward_and_vjp = lambda V, X1, X2: torch.autograd.functional.vjp(
                    lambda X1, X2: kernel_forward(X1, X2), (X1, X2), V
                )
            ctx.kernel_forward_and_vjp = kernel_forward_and_vjp
        if any(ctx.needs_input_grad[1:3]):
            ctx.forward = forward
            ctx.kernel_forward = kernel_forward

        # Done!
        return ST_K_S

    @staticmethod
    def backward(ctx, V):
        """
        O(K^2 I^2) time
        O(I^2 K D) memory
        """
        # Get consistent batch sizes
        X, Sv, Si = ctx.saved_tensors
        K, I = Sv.shape[-2:]
        N, D = X.shape[-2:]
        batch_shape = torch.broadcast_shapes(X.shape[:-2], Sv.shape[:-2])
        X_ = X.expand(*batch_shape, N, D)
        Sv_ = Sv[..., :, None].expand(*batch_shape, K, I, I)
        if Si is not None:
            Si_ = Si[..., :, None].expand(*batch_shape, K, I, I)

        if any(ctx.needs_input_grad[0:2]):
            # Define a vmap function for backward
            # This computes the following 4 gradients:
            #   1) d(v_{1,2} * s_1^T K s_2) / d(s_1)  (... x K)       = v_{1,2} * K s_2
            #   2) d(v_{1,2} * s_1^T K s_2) / d(s_2)  (... x K)       = v_{1,2} * K^T s_1
            #   3) d(v_{1,2} * s_1^T K s_2) / d(X_1)  (... x K x D)   = \sum_{ij} (v{ij} s_{1i} s_{2j}) dK_{ij} / dX_1
            #   4) d(v_{1,2} * s_1^T K s_2) / d(X_2)  (... x K x D)   = \sum_{ij} (v{ij} s_{1i} s_{2j}) dK_{ij} / dX_2
            #
            #   Note: given a VJP for K(X1, X2), the last two terms can be computed as
            #         vjp( K(X1, x2), vector=(v{ij} * s_1 s_2^T) )
            def _sub_backward(
                X1_sub: Float[Tensor, "... K D"],
                X2_sub: Float[Tensor, "... K D"],
                Sv1_sub: Float[Tensor, "... K"],
                Sv2_sub: Float[Tensor, "... K"],
                V_sub: Float[Tensor, "..."],
            ) -> Tuple[Float[Tensor, "... K"], Float[Tensor, "... K D"], Float[Tensor, "... K D"]]:
                vjp_vector = (Sv1_sub[..., :, None] @ Sv2_sub[..., None, :]).mul_(V_sub[..., None, None])
                K, (X1_grad, X2_grad) = ctx.kernel_forward_and_vjp(X1_sub, X2_sub, vjp_vector)
                Sv1_grad = (K @ Sv2_sub[..., None])[..., 0].mul_(V_sub[..., None])
                Sv2_grad = (K.mT @ Sv1_sub[..., None])[..., 0].mul_(V_sub[..., None])
                return Sv1_grad, Sv2_grad, X1_grad, X2_grad

            def _sub_backward_with_index(
                Sv1_sub: Float[Tensor, "... K"],
                Sv2_sub: Float[Tensor, "... K"],
                Si1_sub: Long[LongTensor, "... K"],
                Si2_sub: Long[LongTensor, "... K"],
                V_sub: Float[Tensor, "..."],
            ) -> Float[Tensor, "..."]:
                X1_sub = _vmap_index_select(X_, Si1_sub)
                X2_sub = _vmap_index_select(X_, Si2_sub)
                return _sub_backward(X1_sub, X2_sub, Sv1_sub, Sv2_sub, V_sub)

            # Call d(v_{1,2} s_1^T K s_2) / d... for all 1, 2
            #   1, 2) Sv1_grads, Sv2_grads are (... K x I x I)
            #   3, 4) X1_grads, X2_grads are   (... x K x D x I x I)
            if Si is None:
                sub_backward_fn = _sub_backward
                X_ = X_.view(*batch_shape, I, K, D).transpose(-3, -2).transpose(-2, -1)  # ... x K x D x I
                X_ = X_[..., None].expand(
                    *batch_shape, K, D, I, I
                )  # ... x K x D x I x I, where last every column is identifical
                backward_args = X_, X_.mT, Sv_, Sv_.mT, V
            else:
                sub_backward_fn = _sub_backward_with_index
                backward_args = Sv_, Sv_.mT, Si_, Si_.mT, V
            backward = torch.vmap(
                torch.vmap(sub_backward_fn, in_dims=-1, out_dims=-1),
                in_dims=-1,
                out_dims=-1,
                chunk_size=ctx.chunk_size,
            )
            Sv1_grads, Sv2_grads, X1_grads, X2_grads = backward(*backward_args)

            ########
            # Tricky section of code down below
            # Given `d(v_{1,2} * s_1^T K s_2) / d...` gradients for each 1, 2,
            # we now need to compress these gradients into a single d/dSv tensor and a single d/dX tensor
            ########

            # Compress Sv1_grads and Sv2_grads into a single (... x K x I) tensor
            # Each row of Sv1_grads represents d(v_{1,2} * s_1^T K s_2) / d(s_1) for all s_2
            #    Therefore, we need to sum along all rows
            # Similarly, each column of Sv2_grads represents d(v_{1,2} * s_1^T K s_2) / d(s_2) for all s_1
            #    Therefore, we need to sum along all columns
            Sv_grad = Sv1_grads.sum(dim=-1).add_(Sv2_grads.sum(dim=-2))
            # Compress Sv1_grads and Sv2_grads into a single (... x K x D) tensor
            # We accomplish this compression with the _scatter_gradients helper
            if Si is None:
                X_grad = X1_grads.sum(dim=-1).add_(X2_grads.sum(dim=-2)).transpose(-1, -2).transpose(-2, -3)
                X_grad = X_grad.reshape(*batch_shape, N, D)
            else:
                X_grad = torch.add(
                    _scatter_gradients(X, X1_grads, Si_),
                    _scatter_gradients(X, X2_grads, Si_.mT),
                )
        else:
            X_grad = None
            Sv_grad = None

        return (
            X_grad,
            Sv_grad,
            None,  # Si
            None,  # kernel_forward
            None,  # kernel_vjp
            None,  # chunk_size
        )


class SparseLinearForm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X1: Float[Tensor, "... N D"],
        X2: Float[Tensor, "... N D"],
        Sv: Float[Tensor, "... K I"],
        Si: Long[LongTensor, "... K I"],
        kernel_forward: Callable[[Float[Tensor, "... M D"], Float[Tensor, "... N D"]], Float[Tensor, "... M N"]],
        kernel_forward_and_vjp: Optional[
            Callable[
                [Float[Tensor, "... M N"], Float[Tensor, "... M D"], Float[Tensor, "... N D"]],
                Tuple[Float[Tensor, "... M D"], Float[Tensor, "... N D"]],
            ]
        ],
        chunk_size: Optional[int] = None,
    ):
        """
        X1: tensor of size (m, d)
        X2: tensor of size (n, d)
        Sv: tensor of size (i, nnz), the indices of the sparse entries
        Si: tensor of size (i, nnz), the values of the sparse entries
        kernel_forward: callable function that computes the kernel
        kernel_forward_and_vjp: callable function that computes the kernel forward and vjp

        Return:
            a tensor of size (n, i) representing K(X, X) @ S
        """
        if any(ctx.needs_input_grad[0:2]):
            ctx.save_for_backward(X1, X2, Sv, Si)
            ctx.kernel_forward = kernel_forward
            ctx.kernel_forward_and_vjp = kernel_forward_and_vjp
            ctx.chunk_size = chunk_size

        def _sub_forward(values, indices):
            return kernel_forward(X1, X2[indices]) @ values

        batched_mvm = torch.vmap(_sub_forward, in_dims=0, out_dims=0, chunk_size=chunk_size)
        res = batched_mvm(Sv, Si)
        return res.T

    def backward(ctx, grad_output):
        X1, X2, Sv, Si = ctx.saved_tensors

        grad_output = grad_output.T

        if any(ctx.needs_input_grad[0:2]):
            # Define a vmap function for backward, which computes:
            # 1) d K(X1,X2)S_i / dS_i
            # 2) d K(X1,X2)S_i / dX1
            # 3) d K(X1,X2)S_i / dX2
            def _sub_backward(
                Sv_sub: Float[Tensor, "... K"],
                Si_sub: Long[LongTensor, "... K"],
                V_sub: Float[Tensor, "..."],
            ) -> Tuple[Float[Tensor, "... K"], Float[Tensor, "... K D"], Float[Tensor, "... K D"]]:
                X2_sub = _vmap_index_select(X2, Si_sub)
                vjp_vector = torch.outer(V_sub, Sv_sub)
                K_X1_X2, (X1_grad, X2_grad) = ctx.kernel_forward_and_vjp(X1, X2_sub, vjp_vector)
                Sv_grad = V_sub @ K_X1_X2
                return Sv_grad, X1_grad, X2_grad

            # Call d(v K S) / d...
            Sv_grads, X1_grads, X2_grads = torch.vmap(
                _sub_backward,
                in_dims=0,
                out_dims=0,
                chunk_size=ctx.chunk_size,
            )(Sv, Si, grad_output)

            X1_grad_reduced = X1_grads.sum(dim=0)

            def _reduce_gradient(dx, indices):
                ret = dx.new_zeros(X2.size())
                ret[indices] = dx
                return ret

            X2_grad_reduced = torch.vmap(_reduce_gradient, in_dims=0, out_dims=0, chunk_size=None)(X2_grads, Si)
            # TODO: Should this be chunked for memory efficiency?
        else:
            X1_grad_reduced = None
            X2_grad_reduced = None
            Sv_grads = None

        return (X1_grad_reduced, X2_grad_reduced, Sv_grads, None, None, None, None)
