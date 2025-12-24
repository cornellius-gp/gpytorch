import torch

from linear_operator.operators import DiagLinearOperator, LinearOperator, MatmulLinearOperator
from torch import Tensor

from gpytorch.variational.variational_strategy import VariationalStrategy


class QuadFormDiagonal(torch.autograd.Function):
    r"""A custom autograd function computing the diagonal of a quadratic form.

    This function computes `torch.diag(B' A B)` where `A` is a symmetric matrix. The backward pass saves a large matmul
    compared to PyTorch's default autograd engine when `B` has way more columns than rows.
    """

    @staticmethod
    def forward(ctx, matrix: Tensor, rhs: Tensor):
        r"""The forward pass computing the diagonal of a quadratic form. Note that it does not form `B' A B` explicitly.

        :param matrix: A symmetric matrix of size `(..., M, M)`.
        :param rhs: The right-hand side vector of size `(..., M, N)`.

        :return: The quadratic form diagonal of size `(..., N)`.
        """
        product = matrix @ rhs

        # The backward pass does not need `matrix`
        ctx.save_for_backward(rhs, product)

        return torch.sum(rhs * product, dim=-2)

    @staticmethod
    def backward(ctx, d_diag: Tensor):
        rhs, product = ctx.saved_tensors

        d_matrix = rhs @ (d_diag.unsqueeze(-1) * rhs.mT)
        d_rhs = 2.0 * product * d_diag.unsqueeze(-2)

        return d_matrix, d_rhs


class LargeBatchVariationalStrategy(VariationalStrategy):
    r"""A fast variational strategy implementation optimized for large batch stochastic training on data center GPUs.

    This implementation has two assumptions on the use case:
    1. FP64 operations (in particular triangular solve and matmul) on data center GPUs are not much slower than FP32;
    2. The batch size is very large while the number of inducing points is moderate.

    This implementation speeds up the standard `VariationalStrategy` in two ways:
    1. Group the middle term `K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2}` when computing the predictive covariance, which saves a
    large triangular solve in the forward pass;
    2. Use a custom autograd function computing the diagonal of `K_XZ @ middle_term @ K_ZX` in train mode, which saves
    a large matmul in the backward pass.

    NOTE: Grouping the middle term is not numerically friendly, and thus we have to use double precision to stabilize
    the computation. As a result, this implementation is expected to be slow on CPUs and consumer GPUs. Those who use
    CPUs and consumer cards should use `VariationalStrategy` instead.
    """

    def _compute_predictive_updates(
        self,
        chol: LinearOperator,
        induc_data_covar: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: LinearOperator | None,
        prior_covar: LinearOperator,
        diag: bool = True,
    ) -> tuple[Tensor, LinearOperator]:
        dtype = induc_data_covar.dtype

        # Make `K_ZZ^{-1/2}` dense because `TriangularLinearOperator` does not support solve with `left=False`.
        chol = chol.to_dense().type(torch.float64)

        induc_data_covar = induc_data_covar.type(torch.float64)
        inducing_values = inducing_values.type(torch.float64)

        # The mean update `k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z)`
        inv_chol_t_inducing_values = torch.linalg.solve_triangular(
            chol.mT, inducing_values.unsqueeze(-1), upper=True, left=True
        )
        mean_update = (induc_data_covar.mT @ inv_chol_t_inducing_values).squeeze(-1).type(dtype)

        # The grouped middle term `K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2}`
        middle_term = prior_covar.mul(-1).to_dense()
        if variational_inducing_covar is not None:
            middle_term = variational_inducing_covar.to_dense() + middle_term
        middle_term = middle_term.type(torch.float64)

        middle_term = torch.linalg.solve_triangular(chol, middle_term, upper=False, left=False)
        middle_term = torch.linalg.solve_triangular(chol.mT, middle_term, upper=True, left=True)

        # The covariance update `K_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} K_ZX`
        if diag and self.training:
            # The custom autograd function has a faster backward pass, but it doesn't compute the off-diagonal entries.
            variance_update = QuadFormDiagonal.apply(middle_term, induc_data_covar)
            covar_update = DiagLinearOperator(diag=variance_update.type(dtype))
        else:
            covar_update = MatmulLinearOperator(
                induc_data_covar.mT.type(dtype), (middle_term @ induc_data_covar).type(dtype)
            )

        return mean_update, covar_update
