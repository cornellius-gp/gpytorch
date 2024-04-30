#!/usr/bin/env python3

import types
from typing import Any, Callable, Optional, Tuple

import torch
from torch import LongTensor, Tensor

from ..functions import RBFCovariance
from ..settings import trace_mode
from .kernel import Kernel


class AbstractDtype(type):
    """
    A class that mocks out the behavior of jaxtyping.
    This class allows us to use tensor typehints with sizes.
    https://stackoverflow.com/questions/46382170/how-can-i-create-my-own-parameterized-type-in-python-like-optionalt
    """

    def __getitem__(cls, item: Tuple[Any, str]):
        new_cls = types.new_class(
            f"{cls.__name__}_{item[0].__name__}", (cls,), {}, lambda ns: ns.__setitem__("type", item[0])
        )
        return new_cls


class Float(metaclass=AbstractDtype):
    pass


class Long(metaclass=AbstractDtype):
    pass


def rbf_forward(X1: Float[Tensor, "*batch M D"], X2: Float[Tensor, "*batch N D"]) -> Float[Tensor, "*batch M N"]:
    r"""
    O(NMD) time
    O(NMD) memory

    :param X1: Kernel input :math:`\boldsymbol X_1`
    :param X2: Kernel input :math:`\boldsymbol X_2`
    :return: The RBF kernel matrix :math:`\boldsymbol K_\mathrm{RBF}(\boldsymbol X_1, \boldsymbol X_2)`

    .. note::

        This function does not broadcast. `X1` and `X2` must have the same batch shapes.
    """
    X1_ = X1[..., :, None, :]
    X2_ = X2[..., None, :, :]
    K = (-((X1_ - X2_) ** 2).sum(-1) / 2).exp_()
    return K


def rbf_forward_and_vjp(
    X1: Float[Tensor, "*batch M D"],
    X2: Float[Tensor, "*batch N D"],
    V: Optional[Float[Tensor, "*batch M N"]] = None,
) -> Tuple[Float[Tensor, "*batch M N"], Tuple[Float[Tensor, "*batch M D"], Float[Tensor, "*batch N D"]]]:
    r"""
    O(NMD) time
    O(NMD) memory

    :param X1: Kernel input :math:`\boldsymbol X_1`
    :param X2: Kernel input :math:`\boldsymbol X_2`
    :param V: :math:`\boldsymbol V` - the LHS of the VJP operation
    :return: The kernel matrix :math:`\boldsymbol K` and a tuple containing the VJPs
        :math:`\frac{\del \mathrm{tr} \left( \boldsymbol V^\top \boldsymbol K_\mathrm{RBF}(\boldsymbol X_1, \boldsymbol X_2) \right)}{\del \boldsymbol X_1}`
        and
        :math:`\frac{\del \mathrm{tr} \left( \boldsymbol V^\top \boldsymbol K_\mathrm{RBF}(\boldsymbol X_1, \boldsymbol X_2) \right)}{\del \boldsymbol X_2}`

    .. note::

        This function does not broadcast. `V`, `X1`, and `X2` must have the same batch shapes.
    """  # noqa: E501
    K = rbf_forward(X1, X2)
    VK = (V * K) if V is not None else K
    res = VK[..., None] * (X2[..., None, :, :] - X1[..., :, None, :])
    return K, (res.sum(dim=-2), res.mul(-1).sum(dim=-3))


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
        Si: Long[LongTensor, "... K I"],
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
        batch_shape = torch.broadcast_shapes(X.shape[:-2], Sv.shape[:-2])
        X_ = X.expand(*batch_shape, *X.shape[-2:])
        Sv_ = Sv[..., None, :].expand(*batch_shape, K, I, I)
        Si_ = Si[..., None, :].expand(*batch_shape, K, I, I)

        # Define a vmap function for forward
        # This function essentially computes s_i^T K s_j and s_i^T s_j
        def _sub_forward(
            Sv1_sub: Float[Tensor, "... K"],
            Sv2_sub: Float[Tensor, "... K"],
            Si1_sub: Long[LongTensor, "... K"],
            Si2_sub: Long[LongTensor, "... K"],
        ) -> Float[Tensor, "..."]:
            X1_sub = _vmap_index_select(X_, Si1_sub)
            X2_sub = _vmap_index_select(X_, Si2_sub)
            K_sub = kernel_forward(X1_sub, X2_sub)
            siT_K_sj = (Sv1_sub * (K_sub @ Sv2_sub[..., None]).squeeze(-1)).sum(dim=-1)
            return siT_K_sj

        # Call s_i^T K s_j and s_i^T s_j for all i, j
        forward = torch.vmap(
            torch.vmap(_sub_forward, in_dims=-1, out_dims=-1), in_dims=-1, out_dims=-1, chunk_size=chunk_size
        )
        ST_K_S = forward(Sv_, Sv_.mT, Si_, Si_.mT)

        # Maybe save stuff for the backward pass.
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
        batch_shape = torch.broadcast_shapes(X.shape[:-2], Sv.shape[:-2])
        X_ = X.expand(*batch_shape, *X.shape[-2:])
        Sv_ = Sv[..., None, :].expand(*batch_shape, K, I, I)
        Si_ = Si[..., None, :].expand(*batch_shape, K, I, I)

        # X
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
                Sv1_sub: Float[Tensor, "... K"],
                Sv2_sub: Float[Tensor, "... K"],
                Si1_sub: Long[LongTensor, "... K"],
                Si2_sub: Long[LongTensor, "... K"],
                V_sub: Float[Tensor, "..."],
            ) -> Tuple[Float[Tensor, "... K"], Float[Tensor, "... K D"], Float[Tensor, "... K D"]]:
                X1_sub = _vmap_index_select(X_, Si1_sub)
                X2_sub = _vmap_index_select(X_, Si2_sub)
                vjp_vector = (Sv1_sub[..., :, None] @ Sv2_sub[..., None, :]) * V_sub[..., None, None]
                K, (X1_grad, X2_grad) = ctx.kernel_forward_and_vjp(X1_sub, X2_sub, vjp_vector)
                Sv1_grad = (K @ Sv2_sub[..., None])[..., 0]
                Sv2_grad = (K.mT @ Sv1_sub[..., None])[..., 0]
                return Sv1_grad, Sv2_grad, X1_grad, X2_grad

            # Call d(v_{1,2} s_1^T K s_2) / d... for all 1, 2
            #   1, 2) Sv1_grads, Sv2_grads are (... K x I x I)
            #   3, 4) X1_grads, X2_grads are   (... x K x D x I x I)
            backward = torch.vmap(
                torch.vmap(_sub_backward, in_dims=-1, out_dims=-1),
                in_dims=-1,
                out_dims=-1,
                chunk_size=ctx.chunk_size,
            )
            Sv1_grads, Sv2_grads, X1_grads, X2_grads = backward(Sv_, Sv_.mT, Si_, Si_.mT, V)

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
            Sv_grad = torch.add(
                (Sv1_grads * V[..., None, :, :]).sum(dim=-2),
                (Sv2_grads * V[..., None, :, :]).sum(dim=-1),
            )
            # Compress Sv1_grads and Sv2_grads into a single (... x K x D) tensor
            # We accomplish this compression with the _scatter_gradients helper
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


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class RBFKernel(Kernel):
    r"""
    Computes a covariance matrix based on the RBF (squared exponential) kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{RBF}}(\mathbf{x_1}, \mathbf{x_2}) = \exp \left( -\frac{1}{2}
          (\mathbf{x_1} - \mathbf{x_2})^\top \Theta^{-2} (\mathbf{x_1} - \mathbf{x_2}) \right)
       \end{equation*}

    where :math:`\Theta` is a lengthscale parameter.
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if x1 is a `n x d` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf x1` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `None`)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: `Positive`.)
    :param eps: The minimum value that the lengthscale can take (prevents
        divide by zero errors). (Default: `1e-6`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=5))
        >>> covar = covar_module(x)  # Output: LinearOperator of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])))
        >>> covar = covar_module(x)  # Output: LinearOperator of size (2 x 10 x 10)
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return postprocess_rbf(self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params))
        return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(x1, x2, square_dist=True, diag=False, **params),
        )
