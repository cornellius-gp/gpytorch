#!/usr/bin/env python3

from typing import Tuple

from torch import Tensor

from ..functions import RBFCovariance
from ..settings import trace_mode
from ..typing import Float
from .kernel import Kernel


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

    def _forward(
        self, X1: Float[Tensor, "batch* M D"], X2: Float[Tensor, "batch* N D"]  # noqa F722
    ) -> Float[Tensor, "batch* M N"]:  # noqa F722
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
        diffs = X1_ - X2_
        if diffs.shape[-1] > 1:  # No special casing here causes 10x slowdown!
            dists = diffs.norm(dim=-1)
        else:
            dists = diffs.squeeze(-1)
        K = dists.square_().div_(-2.0).exp_()
        return K

    def _vjp(
        self,
        V: Float[Tensor, "*batch M N"],  # noqa F722
        X1: Float[Tensor, "*batch M D"],  # noqa F722
        X2: Float[Tensor, "*batch N D"],  # noqa F722
    ) -> Tuple[Float[Tensor, "*batch M D"], Float[Tensor, "*batch N D"]]:  # noqa F722
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
        diffs = X1_ - X2_
        if diffs.shape[-1] > 1:  # No special casing here causes 10x slowdown!
            dists = diffs.norm(dim=-1)
        else:
            dists = diffs.squeeze(-1)
        VK = dists.square_().div_(-2.0).exp_().mul_(V)
        res = diffs.mul_(VK[..., None])
        return res.mul(-1).sum(dim=-2), res.sum(dim=-3)
