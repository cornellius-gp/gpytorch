import torch

from ..distributions import MultivariateNormal
from ..lazy import CholLazyTensor, TriangularLazyTensor
from .natural_variational_distribution import (
    AbstractNaturalVariationalDistribution,
    _NaturalToMuVarSqrt,
    _phi_for_cholesky_,
    _triangular_inverse,
)


class TrilNaturalVariationalDistribution(AbstractNaturalVariationalDistribution):
    r"""A multivariate normal :obj:`~gpytorch.variational._VariationalDistribution`,
    parameterized by the natural vector, and a triangular decomposition of the
    natural matrix (which is not the Cholesky).

    If the variational distribution is defined by :math:`\mathcal{N}(\mathbf m, \mathbf S)`, then
    a :obj:`~gpytorch.variational.TrilNaturalVariationalDistribution` uses the parameterization:

    .. math::

        \begin{align*}
            \boldsymbol \theta_\text{vec} &= \mathbf S^{-1} \mathbf m
            \\
            \boldsymbol \Theta_\text{tril\_mat} &= \text{cholesky}(\mathbf S)^{-1}
        \end{align*}

    The parameter :math:`\boldsymbol \Theta_\text{tril\_mat}`
    (`natural_tril_mat`) is a lower triangular matrix.

    The gradients with respect to the variational parameters calculated by this
    class are instead the **natural** gradients. Thus, optimising its
    parameters using gradient descent (:obj:`~torch.optim.SGDOptimizer`)
    becomes natural gradient descent (see e.g. `Salimbeni et al., 2018`_).

    .. seealso::
        The `natural gradient descent tutorial
        <examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.ipynb>`_
        for use instructions.

        The :obs:`~gpytorch.variational.NaturalVariationalDistribution`, which
        needs less iterations to make variational regression converge, at the
        cost of introducing numerical instability.

    .. note::
        The relationship of the parameter :math:`\boldsymbol \Theta_\text{tril\_mat}`
        to the natural parameter :math:`\boldsymbol \Theta_\text{mat}` from
        :obj:`~gpytorch.variational.NaturalVariationalDistribution` is
        :math:`\boldsymbol \Theta_\text{mat} = -1/2 {\boldsymbol \Theta_\text{tril\_mat}}^T
               {\boldsymbol \Theta_\text{tril\_mat}}`.
        Note that this is not the form of the Cholesky decomposition of :math:`\boldsymbol \Theta_\text{mat}`.

    .. _Salimbeni et al., 2018:
        https://arxiv.org/abs/1803.09151

    :param int num_inducing_points: Size of the variational distribution. This implies that the variational mean
        should be this size, and the variational covariance matrix should have this many rows and columns.
    :param batch_shape: Specifies an optional batch size
        for the variational parameters. This is useful for example when doing additive variational inference.
    :type batch_shape: :obj:`torch.Size`, optional
    :param float mean_init_std: (Default: 1e-3) Standard deviation of gaussian noise to add to the mean initialization.

    """

    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3, **kwargs):
        super().__init__(num_inducing_points=num_inducing_points, batch_shape=batch_shape, mean_init_std=mean_init_std)
        scaled_mean_init = torch.zeros(num_inducing_points)
        neg_prec_init = torch.eye(num_inducing_points, num_inducing_points)
        scaled_mean_init = scaled_mean_init.repeat(*batch_shape, 1)
        neg_prec_init = neg_prec_init.repeat(*batch_shape, 1, 1)

        # eta1 and tril_dec(eta2) parameterization of the variational distribution
        self.register_parameter(name="natural_vec", parameter=torch.nn.Parameter(scaled_mean_init))
        self.register_parameter(name="natural_tril_mat", parameter=torch.nn.Parameter(neg_prec_init))

    def forward(self):
        mean, chol_covar = _TrilNaturalToMuVarSqrt.apply(self.natural_vec, self.natural_tril_mat)
        return MultivariateNormal(mean, CholLazyTensor(TriangularLazyTensor(chol_covar)))

    def initialize_variational_distribution(self, prior_dist):
        prior_cov = prior_dist.lazy_covariance_matrix
        chol = prior_cov.cholesky().evaluate()
        tril_mat = _triangular_inverse(chol, upper=False)

        natural_vec = prior_cov.inv_matmul(prior_dist.mean.unsqueeze(-1)).squeeze(-1)
        noise = torch.randn_like(natural_vec).mul_(self.mean_init_std)

        self.natural_vec.data.copy_(natural_vec.add_(noise))
        self.natural_tril_mat.data.copy_(tril_mat)


class _TrilNaturalToMuVarSqrt(torch.autograd.Function):
    @staticmethod
    def _forward(nat_mean, tril_nat_covar):
        L = _triangular_inverse(tril_nat_covar, upper=False)
        mu = L @ (L.transpose(-1, -2) @ nat_mean.unsqueeze(-1))
        return mu.squeeze(-1), L
        # return nat_mean, L

    @staticmethod
    def forward(ctx, nat_mean, tril_nat_covar):
        mu, L = _TrilNaturalToMuVarSqrt._forward(nat_mean, tril_nat_covar)
        ctx.save_for_backward(mu, L, tril_nat_covar)
        return mu, L

    @staticmethod
    def backward(ctx, dout_dmu, dout_dL):
        mu, L, C = ctx.saved_tensors
        dout_dnat1, dout_dnat2 = _NaturalToMuVarSqrt._backward(dout_dmu, dout_dL, mu, L, C)
        """
        Now we need to do the Jacobian-Vector Product for the transformation:
        L = inv(chol(inv(-2 theta_cov)))

        C^T C = -2theta_cov

        so we need to do forward differentiation, starting with sensitivity:
        theta_cov = dout_dnat2

        and ending with sensitivity _C

        if B = inv(-2 theta_cov) then:

        _B  =  d inv(-2 theta_cov)/dtheta_cov * _theta_cov  =  -B (-2 _theta_cov) B

        if L = chol(B), B = LL^T then (https://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf):

        _L = L phi(L^{-1} _B_ (L^{-1})^T) = L phi(2 L^T _theta_cov L)

        Then C = inv(L), so

        _C = -C _L C = phi(-2 L^T _theta_cov L)C
        """
        A = L.transpose(-2, -1) @ dout_dnat2 @ L
        phi = _phi_for_cholesky_(A.mul_(-2))
        dout_dtril = phi @ C
        return dout_dnat1, dout_dtril
