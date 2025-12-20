import torch

from linear_operator.operators import LinearOperator, MatmulLinearOperator
from torch import Tensor

from gpytorch.variational.variational_strategy import VariationalStrategy


class LargeBatchVariationalStrategy(VariationalStrategy):
    r"""A fast variational strategy implementation optimized for large batch stochastic training on data center GPUs.

    This implementation has two assumptions:
    1. FP64 operations (in particular triangular solve and matmul) on data center GPUs are almost as fast as FP32;
    2. The batch size is very large while the number of inducing points is moderate.

    The main idea is to group the middle term `K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2}` when computing the predictive
    covariance. The caveat is that grouping the middle term in this way is not numerically friendly, and thus we have
    to use double precision to stabilize the computation.

    NOTE: This implementation does tensor operations in double precision. Thus, it will be slow on CPUs and consumer
    GPUs. Those who use CPUs and consumer cards should use `VariationalStrategy` instead.
    """

    def _compute_predictive_updates(
        self,
        chol: LinearOperator,
        induc_data_covar: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: LinearOperator | None,
        prior_covar: LinearOperator,
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
        covar_update = MatmulLinearOperator(
            induc_data_covar.mT.type(dtype), (middle_term @ induc_data_covar).type(dtype)
        )

        return mean_update, covar_update
