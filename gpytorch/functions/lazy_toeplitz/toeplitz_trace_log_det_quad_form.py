import torch
from torch.autograd import Function
from gpytorch.utils.toeplitz import toeplitz_mv, toeplitz_mm, sym_toeplitz_derivative_quadratic_form
from gpytorch.utils import LinearCG, StochasticLQ


class ToeplitzTraceLogDetQuadForm(Function):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def forward(self, mu_diff, chol_covar1, covar2_toeplitz_column):
        def mv_closure(v):
            return toeplitz_mv(covar2_toeplitz_column, covar2_toeplitz_column, v)

        def quad_form_closure(z):
            return z.dot(LinearCG().solve(mv_closure, chol_covar1.t().mv(chol_covar1.mv(z))))

        # log |K2|
        log_det_covar2, = StochasticLQ(num_random_probes=10).evaluate(mv_closure,
                                                                      len(mu_diff),
                                                                      [lambda x: x.log()])

        sample_matrix = torch.sign(torch.randn(self.num_samples, len(covar2_toeplitz_column)))

        # Tr(K2^{-1}K1)
        trace = 0
        for z in sample_matrix:
            trace = trace + quad_form_closure(z)
        trace = trace / self.num_samples

        mat_inv_y = LinearCG().solve(mv_closure, mu_diff)

        # Inverse quad form
        inv_quad_form = mat_inv_y.dot(mu_diff)

        res = log_det_covar2 + trace + inv_quad_form

        self.save_for_backward(mu_diff, chol_covar1, covar2_toeplitz_column)
        self.mv_closure = mv_closure
        self.mat_inv_y = mat_inv_y

        return mu_diff.new().resize_(1).fill_(res)

    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]

        mu_diff, chol_covar1, covar2_toeplitz_column = self.saved_tensors
        mat_inv_y = self.mat_inv_y
        mv_closure = self.mv_closure

        def deriv_quad_form_closure(z):
            I_minus_Tinv_M_z = z - LinearCG().solve(mv_closure, chol_covar1.t().mv(chol_covar1.mv(z)))
            Tinv_z = LinearCG().solve(mv_closure, z)
            return sym_toeplitz_derivative_quadratic_form(Tinv_z, I_minus_Tinv_M_z)

        def toeplitz_mm_closure(M):
            return toeplitz_mm(covar2_toeplitz_column, covar2_toeplitz_column, M)

        grad_mu_diff = None
        grad_cholesky_factor = None
        grad_toeplitz_column = None

        if self.needs_input_grad[0]:
            # Need gradient with respect to mu_diff
            grad_mu_diff = mat_inv_y.mul_(2 * grad_output_value)

        if self.needs_input_grad[1]:
            # Compute gradient with respect to the Cholesky factor L
            grad_cholesky_factor = 2 * LinearCG().solve(toeplitz_mm_closure, chol_covar1)
            grad_cholesky_factor.mul_(grad_output_value)

        if self.needs_input_grad[2]:
            sample_matrix = torch.sign(torch.randn(self.num_samples, len(covar2_toeplitz_column)))

            grad_toeplitz_column = torch.zeros(len(covar2_toeplitz_column))
            for z in sample_matrix:
                grad_toeplitz_column = grad_toeplitz_column + deriv_quad_form_closure(z)
            grad_toeplitz_column = grad_toeplitz_column / self.num_samples

            grad_toeplitz_column = grad_toeplitz_column - sym_toeplitz_derivative_quadratic_form(mat_inv_y.squeeze(),
                                                                                                 mat_inv_y.squeeze())
            grad_toeplitz_column.mul_(grad_output_value)

        return grad_mu_diff, grad_cholesky_factor, grad_toeplitz_column
