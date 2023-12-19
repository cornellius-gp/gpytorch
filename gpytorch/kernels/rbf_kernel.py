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


def rbf_forward(X1: Float[Tensor, "batch* M D"], X2: Float[Tensor, "batch* N D"]) -> Float[Tensor, "batch* M N"]:
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
    K = (-((X1_ - X2_) ** 2).sum(-1) / 2).exp()
    return K


def rbf_vjp(
    V: Float[Tensor, "*batch M N"],
    X1: Float[Tensor, "*batch M D"],
    X2: Float[Tensor, "*batch N D"],
) -> Tuple[Float[Tensor, "*batch M D"], Float[Tensor, "*batch N D"]]:
    r"""
    O(NMD) time
    O(NMD) memory

    :param V: :math:`\boldsymbol V` - the LHS of the VJP operation
    :param X1: Kernel input :math:`\boldsymbol X_1`
    :param X2: Kernel input :math:`\boldsymbol X_2`
    :return: The VJPs
        :math:`\frac{\del \mathrm{tr} \left( \boldsymbol V^\top \boldsymbol K_\mathrm{RBF}(\boldsymbol X_1, \boldsymbol X_2) \right)}{\del \boldsymbol X_1}`
        and
        :math:`\frac{\del \mathrm{tr} \left( \boldsymbol V^\top \boldsymbol K_\mathrm{RBF}(\boldsymbol X_1, \boldsymbol X_2) \right)}{\del \boldsymbol X_2}`

    .. note::

        This function does not broadcast. `V`, `X1`, and `X2` must have the same batch shapes.
    """  # noqa: E501
    X1_ = X1[..., :, None, :]
    X2_ = X2[..., None, :, :]
    VK = V * (-((X1_ - X2_) ** 2).sum(-1) / 2).exp()
    res = VK[..., None] * (X2_ - X1_)
    return res.sum(dim=-2), res.mul(-1).sum(dim=-3)


def _compress_gradients(
    X: Float[Tensor, "... N D"],
    X_grads_unreduced: Float[Tensor, "*batch K I I D"],
    Si: Long[LongTensor, "*batch K I I"],
):
    batch_shape = X_grads_unreduced.shape[:-4]
    N, D = X.shape[-2:]
    K, I = X_grads_unreduced.shape[-4:-2]

    # Prepare X_grads_unreduced and Si for a vmap
    X_grads_unreduced = X_grads_unreduced.reshape(*batch_shape, K * I * I, D)  # *batch x KI^2 x D
    Si = Si.reshape(*batch_shape, -1)  # *batch x KI^2

    # Function to reduce X_grads_unreduced
    def reduce_grad(X_grads_unreduced_sub: Float[Tensor, "... K*I*I"]):
        res = torch.zeros(*batch_shape, N)
        print("in vmap", res.shape, X_grads_unreduced_sub.shape, Si.shape)
        res = torch.scatter_reduce(res, -1, Si, X_grads_unreduced_sub, reduce="sum")
        return res

    # Run vmap to reduce X_grads_unreduced
    X_grad = torch.vmap(reduce_grad, in_dims=-1, out_dims=-1)(X_grads_unreduced)

    # Now compress expanded batch dimensions
    _, X_grad = torch.autograd.functional.vjp(lambda _X: _X.expand(X_grad.shape), X, X_grad)
    return X_grad


def _vmap_index_select(
    input: Float[Tensor, "*batch M N"],
    index: Float[Tensor, "*batch K"],
) -> Float[Tensor, "*batch K L"]:
    """ """
    if input.dim() == 2:
        assert index.dim() == 1
        return torch.index_select(input=input, dim=-2, index=index)
    else:
        return torch.vmap(_vmap_index_select)(input, index)


class SparseQuadForm(torch.autograd.Function):
    r"""
    An autograd function to compute quadratic forms
    .. math::

        \boldsymbol S_1^T \boldsymbol K(\boldsymbol X_1, \boldsymbol X_2) S_2^\top,

    where :math:`\boldsymbol S_1` and :math:`\boldsymbol S_2` are row-wise sparse.
    They are represented by row-wise value/index matrices.

    The kernel matrix is given by a forward (and an optional backward) function.
    """

    @staticmethod
    def forward(
        ctx,
        X1: Float[Tensor, "... M D"],
        X2: Float[Tensor, "... N D"],
        Sv1: Float[Tensor, "... K I"],
        Sv2: Float[Tensor, "... K I"],
        Si1: Long[LongTensor, "... K I"],
        Si2: Long[LongTensor, "... K I"],
        kernel_forward: Callable[[Float[Tensor, "... M D"], Float[Tensor, "... N D"]], Float[Tensor, "... M N"]],
        kernel_vjp: Optional[
            Callable[
                [Float[Tensor, "... M N"], Float[Tensor, "... M D"], Float[Tensor, "... N D"]],
                Tuple[Float[Tensor, "... M D"], Float[Tensor, "... N D"]],
            ]
        ],
    ) -> Float[Tensor, "... I I"]:
        """
        O(K^2 I^2) time
        O(max(K^2, I^2)) memory (if vmap is done sequentially)
        """

        ctx.kernel_vjp = kernel_vjp

        # Get consistent batch sizes
        K, I = Sv1.shape[-2:]
        batch_shape = torch.broadcast_shapes(
            X1.shape[:-2], X2.shape[:-2], Sv1.shape[:-2], Sv2.shape[:-2], Si1.shape[:-2], Si2.shape[:-2]
        )
        X1_ = X1.expand(*batch_shape, *X1.shape[-2:])
        X2_ = X2.expand(*batch_shape, *X2.shape[-2:])
        Sv1_ = Sv1[..., :, None].expand(*batch_shape, K, I, I)
        Sv2_ = Sv2[..., None, :].expand(*batch_shape, K, I, I)
        Si1_ = Si1[..., :, None].expand(*batch_shape, K, I, I)
        Si2_ = Si2[..., None, :].expand(*batch_shape, K, I, I)

        # Define a vmap function for forward
        # This function essentially computes s_i^T K s_j
        def _sub_forward(
            Sv1_sub: Float[Tensor, "... K"],
            Sv2_sub: Float[Tensor, "... K"],
            Si1_sub: Long[LongTensor, "... K"],
            Si2_sub: Long[LongTensor, "... K"],
        ) -> Float[Tensor, "... K K"]:
            X1_sub = _vmap_index_select(X1_, Si1_sub)
            X2_sub = _vmap_index_select(X2_, Si2_sub)
            K_sub = kernel_forward(X1_sub, X2_sub)
            res = (Sv1_sub * (K_sub @ Sv2_sub[..., None]).squeeze(-1)).sum(dim=-1)
            return res

        # Call s_i^T K s_j for all i, j
        forward = torch.vmap(torch.vmap(_sub_forward, in_dims=-1, out_dims=-1), in_dims=-1, out_dims=-1)
        res = forward(Sv1_, Sv2_, Si1_, Si2_)

        # Done!
        ctx.save_for_backward(X1, X2, Sv1, Sv2, Si1, Si2)
        return res

    @staticmethod
    def backward(ctx, V):
        """
        O(K^2 I^2) time
        O(max(I^2 K D) memory
        """
        kernel_vjp = ctx.kernel_vjp
        if kernel_vjp is None:
            kernel_vjp = lambda V, X1, X2: torch.func.vjp(lambda X1, X2: ctx.kernel_forward(X1, X2), (X1, X2), V)[1]

        # Get consistent batch sizes
        X1, X2, Sv1, Sv2, Si1, Si2 = ctx.saved_tensors
        K, I = Sv1.shape[-2:]
        batch_shape = torch.broadcast_shapes(
            X1.shape[:-2], X2.shape[:-2], Sv1.shape[:-2], Sv2.shape[:-2], Si1.shape[:-2], Si2.shape[:-2]
        )
        X1_ = X1.expand(*batch_shape, *X1.shape[-2:])
        X2_ = X2.expand(*batch_shape, *X2.shape[-2:])
        Sv1_ = Sv1[..., :, None].expand(*batch_shape, K, I, I)
        Sv2_ = Sv2[..., None, :].expand(*batch_shape, K, I, I)
        Si1_ = Si1[..., :, None].expand(*batch_shape, K, I, I)
        Si2_ = Si2[..., None, :].expand(*batch_shape, K, I, I)

        if any(ctx.needs_input_grad[:-2]):
            # Define a vmap function for backward
            # This function essentially computes d(v_ij * s_i^T K s_j) / d(x_1), and likewise for x_2
            def _sub_backward(
                V_sub: Float[Tensor, "..."],
                Sv1_sub: Float[Tensor, "... K"],
                Sv2_sub: Float[Tensor, "... K"],
                Si1_sub: Long[LongTensor, "... K"],
                Si2_sub: Long[LongTensor, "... K"],
            ) -> Float[Tensor, "... K K"]:
                X1_sub = _vmap_index_select(X1_, Si1_sub)
                X2_sub = _vmap_index_select(X2_, Si2_sub)
                X1_sub_grad, X2_sub_grad = kernel_vjp(
                    V_sub[..., None, None] * (Sv1_sub[..., :, None] @ Sv2_sub[..., None, :]), X1_sub, X2_sub
                )
                print(X1_sub_grad.shape, X2_sub_grad.shape)
                return torch.cat([X1_sub_grad, X2_sub_grad], dim=-2)  # Vmap can only return single tensors

            # Call s_i^T K s_j for all i, j
            backward = torch.vmap(torch.vmap(_sub_backward, in_dims=-1, out_dims=-2), in_dims=-1, out_dims=-2)
            combined_grads = backward(V, Sv1_, Sv2_, Si1_, Si2_)  # ... x k*2 x i x i x d
            X1_grads, X2_grads = torch.split(combined_grads, [K, K], dim=-4)
            X1_grad = _compress_gradients(X1, X1_grads, Si1_)
            X2_grad = _compress_gradients(X2, X2_grads, Si2_)
        else:
            X1_grad = None
            X2_grad = None

        return (
            X1_grad,
            X2_grad,
            None,  # Sv1
            None,  # Sv2
            None,  # Si1
            None,  # Si2
            None,  # kernel_forward
            None,  # kernel_vjp
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
