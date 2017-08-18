import torch
from torch.autograd import Function
from gpytorch.utils import LinearCG, StochasticLQ


class TraceLogDetQuadForm(Function):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def forward(self, mu_diff, chol_covar1, covar2):
        mul_closure = covar2

        def quad_form_closure(z):
            return z.dot(LinearCG().solve(mul_closure, chol_covar1.t().mv(chol_covar1.mv(z))))

        # log |K2|
        log_det_covar2, = StochasticLQ(num_random_probes=10).evaluate(mul_closure, len(mu_diff), [lambda x: x.log()])

        # Tr(K2^{-1}K1)
        sample_matrix = torch.sign(torch.randn(self.num_samples, len(mu_diff)))
        trace = 0
        for z in sample_matrix:
            trace = trace + quad_form_closure(z)
        trace = trace / self.num_samples

        # Inverse quad form
        mat_inv_y = LinearCG().solve(mul_closure, mu_diff)
        inv_quad_form = mat_inv_y.dot(mu_diff)

        res = log_det_covar2 + trace + inv_quad_form

        self.save_for_backward(mu_diff, chol_covar1, covar2)
        self.mul_closure = mul_closure
        self.mat_inv_y = mat_inv_y

        return mu_diff.new().resize_(1).fill_(res)

    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]

        mu_diff, chol_covar1, covar2 = self.saved_tensors
        mat_inv_y = self.mat_inv_y
        mul_closure = self.mul_closure

        grad_mu_diff = None
        grad_cholesky_factor = None
        grad_covar2 = None

        if self.needs_input_grad[0]:
            # Need gradient with respect to mu_diff
            grad_mu_diff = mat_inv_y.mul_(2 * grad_output_value)

        if self.needs_input_grad[1]:
            # Compute gradient with respect to the Cholesky factor L
            grad_cholesky_factor = 2 * LinearCG().solve(mul_closure, chol_covar1)
            grad_cholesky_factor.mul_(grad_output_value)

        if self.needs_input_grad[2]:
            # Compute gradient with respect to covar2
            mul_t_closure = covar2.t()
            covar2_t_inv_mu = LinearCG().solve(mul_t_closure, mu_diff)
            covar2_inv_mu = LinearCG().solve(mul_closure, mu_diff)
            quad_part = covar2_t_inv_mu.unsqueeze(1).mm(covar2_inv_mu.unsqueeze(0))

            sample_matrix = torch.sign(torch.randn(self.num_samples, len(mu_diff)))
            logdet_trace_part = torch.zeros(covar2.size())

            def deriv_quad_form_closure(z):
                I_minus_covar2inv_covar1_z = z - LinearCG().solve(mul_closure, chol_covar1.t().mv(chol_covar1.mv(z)))
                covar2_t_inv_z = LinearCG().solve(mul_t_closure, z)
                return covar2_t_inv_z.unsqueeze(1).mm(I_minus_covar2inv_covar1_z.unsqueeze(0))

            for z in sample_matrix:
                logdet_trace_part = logdet_trace_part + deriv_quad_form_closure(z)

            logdet_trace_part.div_(self.num_samples)

            grad_covar2 = quad_part + logdet_trace_part
            grad_covar2 = grad_covar2.mul_(grad_output_value)

        return grad_mu_diff, grad_cholesky_factor, grad_covar2
