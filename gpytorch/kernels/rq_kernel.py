from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Optional

import torch

from ..constraints import Interval, Positive
from .kernel import Kernel


class RQKernel(Kernel):
    r"""
    Computes a covariance matrix based on the rational quadratic kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{RQ}}(\mathbf{x_1}, \mathbf{x_2}) =  \left(1 + \frac{1}{2\alpha}
          (\mathbf{x_1} - \mathbf{x_2})^\top \Theta^{-2} (\mathbf{x_1} - \mathbf{x_2}) \right)^{-\alpha}
       \end{equation*}

    where :math:`\Theta` is a lengthscale parameter, and :math:`\alpha` is the
    rational quadratic relative weighting parameter.
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
    :param alpha_constraint:
        Set this if you want to apply a constraint to the alpha parameter. (Default: `Positive`.)
    :param eps: The minimum value that the lengthscale can take (prevents
        divide by zero errors). (Default: `1e-6`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.
    :ivar torch.Tensor alpha: The rational quadratic relative weighting parameter. Size/shape of parameter depends
        on the batch_shape argument
    """

    has_lengthscale = True

    def __init__(self, alpha_constraint: Optional[Interval] = None, **kwargs):
        super(RQKernel, self).__init__(**kwargs)
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        if alpha_constraint is None:
            alpha_constraint = Positive()

        self.register_constraint("raw_alpha", alpha_constraint)

    def forward(self, x1, x2, diag=False, **params):
        def postprocess_rq(dist):
            alpha = self.alpha
            for _ in range(1, len(dist.shape) - len(self.batch_shape)):
                alpha = alpha.unsqueeze(-1)
            return (1 + dist.div(2 * alpha)).pow(-alpha)

        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        return self.covar_dist(
            x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rq, postprocess=True, **params
        )

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))
