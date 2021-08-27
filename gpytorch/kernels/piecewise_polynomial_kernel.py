import torch

from .kernel import Kernel


class PiecewisePolynomialKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Piecewise Polynomial kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

        \begin{equation*}
            r = \norm{x1 - x2}
            j = \lfloor frac{D}{2} \rfloor + q +1
            K_{\text{ppD, 0}}(\mathbf{x_1}, \mathbf{x_2}) = (1-r)^j_+ ,
            K_{\text{ppD, 1}}(\mathbf{x_1}, \mathbf{x_2}) = (1-r)^{j+1}_+ ((j + 1)r + 1),
            K_{\text{ppD, 2}}(\mathbf{x_1}, \mathbf{x_2}) = (1-r)^{j+2}_+ ((1 + (j+2)r +
                \frac{j^2 + 4j + 3}{3}r^2),
            K_{\text{ppD, 3}}(\mathbf{x_1}, \mathbf{x_2}) = (1-r)^{j+3}_+
                (1 + (j+3)r + \frac{6j^2 + 36j + 45}{15}r^2 +
                \frac{j^3 + 9j^2 + 23j +15}{15}r^3)
        \end{equation*}

    where :math: `K_{\text{ppD, q}}` is positive semidefinite in :math: `\mathbb{R}^{D}` and
    :math: `q` is the smoothness coefficient. Equation taken from Equation 4.21, C. E. Rasmussen & C. K. I. Williams,
    Gaussian Processes for Machine Learning, the MIT Press, 2006, ISBN 026218253X.
    c 2006 Massachusetts Institute of Technology. www.GaussianProcess.org/gpml

        .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    :param q: (default= 2) The smoothness parameter.
    :type q: int (0, 1, 2 or 3)
    :param int ard_num_dims: (optional, default=None) Set this if you want a separate lengthscale for each
        input dimension. It should be `d` if `x1` is a `n x d` matrix. Default: `None`
    :param batch_shape: (optional, default = torch.Size([])) Set this if you want a separate
        lengthscale for each batch of input data. It should be `b` if `x1` is a `b x n x d` tensor.
    :type batch_shape: torch.Size([])
    :param active_dims: (optional, default = None)Set this if you want to compute the
        covariance of only a few input dimensions. The ints corresponds to the indices of the
        dimensions.
    :type active_dims: tuple of ints
    :param lengthscale_prior: (optional, default = None) Set this if you want to apply
        a prior to the lengthscale parameter.
    :type lengthscale_prior: ~gpytorch.priors.Prior
    :param lengthscale_constraint: (optional, default = Positive) Set this if you want to
        apply a constraint to the lengthscale parameter.
    :type lengthscale_constraint: ~gpytorch.constraints.Positive
    :param eps (float): (Default= 1e-6): The minimum value that the lengthscale can take
        (prevents divide by zero errors).

    Attributes:
        :param:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :param:`ard_num_dims` and :param:`batch_shape` arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch option
        >>> covar_module = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.PiecewisePolynomialKernel(q = 2))
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.PiecewisePolynomialKernel(q = 2, ard_num_dims=5)
                            )
        >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PiecewisePolynomialKernel(q = 2, batch_shape=torch.Size([2]))
            )
        >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """
    has_lengthscale = True

    def __init__(self, q=2, **kwargs):
        super(PiecewisePolynomialKernel, self).__init__(**kwargs)
        if q not in {0, 1, 2, 3}:
            raise ValueError("q expected to be 0, 1, 2 or 3")
        self.q = q

    def fmax(self, r, j, q):
        return torch.max(torch.tensor(0.0), 1 - r).pow(j + q)

    def get_cov(self, r, j, q):
        if q == 0:
            return 1
        if q == 1:
            return (j + 1) * r + 1
        if q == 2:
            return 1 + (j + 2) * r + ((j ** 2 + 4 * j + 3) / 3.0) * r ** 2
        if q == 3:
            return (
                1
                + (j + 3) * r
                + ((6 * j ** 2 + 36 * j + 45) / 15.0) * r ** 2
                + ((j ** 3 + 9 * j ** 2 + 23 * j + 15) / 15.0) * r ** 3
            )

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        if last_dim_is_batch is True:
            D = x1.shape[1]
        else:
            D = x1.shape[-1]
        j = torch.floor(torch.tensor(D / 2.0)) + self.q + 1
        if last_dim_is_batch and diag:
            r = self.covar_dist(x1_, x2_, last_dim_is_batch=True, diag=True)
        elif diag:
            r = self.covar_dist(x1_, x2_, diag=True)
        elif last_dim_is_batch:
            r = self.covar_dist(x1_, x2_, last_dim_is_batch=True)
        else:
            r = self.covar_dist(x1_, x2_)
        cov_matrix = self.fmax(r, j, self.q) * self.get_cov(r, j, self.q)
        return cov_matrix