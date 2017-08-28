from torch.autograd import Function
from .lincg import LinearCG
from .lanczos_quadrature import StochasticLQ
import torch
import math


def _default_mm_closure_factor(x):
    return x


def _default_grad_fn(grad_output, rhs_mat):
    return rhs_mat.t().mm(grad_output),


def _default_derivative_quadratic_form_factory(mat):
    return lambda left_vector, right_vector: (left_vector.unsqueeze(1).mm(right_vector.unsqueeze(0)),)


def _default_exact_gp_mml_grad_closure_factory(*args):
    def closure(mm_closure, tr_inv, mat_inv_labels, labels, num_samples):
        grad = torch.ger(labels.view(-1), mat_inv_labels.view(-1))
        grad.add_(-torch.eye(*grad.size()))
        grad = LinearCG().solve(mm_closure, grad)
        return grad,
    return closure


def invmm_factory(mm_closure_factory=_default_mm_closure_factor, grad_fn=_default_grad_fn):
    class Invmm(Function):
        def __init__(self, *args):
            self.args = args

        def forward(self, *args):
            closure_args = self.args + args[:-1]
            rhs_matrix = args[-1]
            res = LinearCG().solve(mm_closure_factory(*closure_args), rhs_matrix)
            if res.ndimension() == 1:
                res.unsqueeze_(1)
            self.save_for_backward(*(list(args) + [res]))
            return res

        def backward(self, grad_output):
            if grad_fn is None:
                raise NotImplementedError

            closure_args = self.args + self.saved_tensors[:-2]
            input_1_t_input_2 = self.saved_tensors[-1]

            closure_arg_grads = [None] * len(closure_args)
            rhs_matrix_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                lhs_matrix_grad = LinearCG().solve(mm_closure_factory(*closure_args), grad_output)
                if lhs_matrix_grad.ndimension() == 1:
                    lhs_matrix_grad.unsqueeze_(1)
                lhs_matrix_grad = lhs_matrix_grad.mul_(-1)
                closure_arg_grads = list(grad_fn(input_1_t_input_2.t(), lhs_matrix_grad.t()))

            # input_2 gradient
            if self.needs_input_grad[-1]:
                rhs_matrix_grad = LinearCG().solve(mm_closure_factory(*closure_args), grad_output)

            return tuple(closure_arg_grads + [rhs_matrix_grad])

    return Invmm


def mm_factory(mm_closure_factory=_default_mm_closure_factor, grad_fn=_default_grad_fn):
    class Mm(Function):
        def __init__(self, *args):
            self.args = args

        def forward(self, *args):
            closure_args = self.args + args[:-1]
            rhs_matrix = args[-1]
            res = mm_closure_factory(*closure_args)(rhs_matrix)
            if res.ndimension() == 1:
                res.unsqueeze_(1)
            self.save_for_backward(*(list(args) + [res]))
            return res

        def backward(self, grad_output):
            if grad_fn is None:
                raise NotImplementedError

            closure_args = self.args + self.saved_tensors[:-2]
            input_1_t_input_2 = self.saved_tensors[-1]

            closure_arg_grads = [None] * len(closure_args)
            rhs_matrix_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                closure_arg_grads = list(grad_fn(grad_output, input_1_t_input_2))

            # input_2 gradient
            if self.needs_input_grad[-1]:
                rhs_matrix_grad = mm_closure_factory(*closure_args)(grad_output)

            return tuple(closure_arg_grads + [rhs_matrix_grad])

    return Mm


def trace_logdet_quad_form_factory(mm_closure_factory=_default_mm_closure_factor,
                                   derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    """
    Args:
        - covar2_mm - covar2_mv(matrix, *covar2_args) = covar2 * matrix
        - derivative_quadratic_form - derivative_quadratic_form(left_vector, right_vector, *covar2_args)
                                      = \delta left_vector * covar2 * right_vector / \delta covar2_args
                                      The result of this function should be a list consistent to *covar2_args.
    """
    class TraceLogDetQuadForm(Function):
        def __init__(self, num_samples=10):
            self.num_samples = num_samples

        def forward(self, mu_diff, chol_covar1, *covar2_args):
            if isinstance(mm_closure_factory(*covar2_args), torch.Tensor):
                covar2_mv_closure = mm_closure_factory(*covar2_args)
            else:
                def covar2_mv_closure(vector):
                    return mm_closure_factory(*covar2_args)(vector.unsqueeze(1)).squeeze()

            def quad_form_closure(z):
                return z.dot(LinearCG().solve(covar2_mv_closure, chol_covar1.t().mv(chol_covar1.mv(z))))

            # log |K2|
            log_det_covar2, = StochasticLQ(num_random_probes=10).evaluate(covar2_mv_closure,
                                                                          len(mu_diff),
                                                                          [lambda x: x.log()])

            # Tr(K2^{-1}K1)
            sample_matrix = torch.sign(torch.randn(self.num_samples, len(mu_diff)))
            trace = 0
            for z in sample_matrix:
                trace = trace + quad_form_closure(z)
            trace = trace / self.num_samples

            # Inverse quad form
            mat_inv_y = LinearCG().solve(covar2_mv_closure, mu_diff)
            inv_quad_form = mat_inv_y.dot(mu_diff)

            res = log_det_covar2 + trace + inv_quad_form

            self.save_for_backward(*([mu_diff] + [chol_covar1] + list(covar2_args)))
            self.covar2_mv_closure = covar2_mv_closure
            self.mat_inv_y = mat_inv_y

            return mu_diff.new().resize_(1).fill_(res)

        def backward(self, grad_output):
            grad_output_value = grad_output.squeeze()[0]

            args = self.saved_tensors

            mu_diff = args[0]
            chol_covar1 = args[1]
            covar2_args = args[2:]

            mat_inv_y = self.mat_inv_y
            covar2_mv_closure = self.covar2_mv_closure

            grad_mu_diff = None
            grad_cholesky_factor = None
            grad_covar2_args = [None] * len(covar2_args)

            if self.needs_input_grad[0]:
                # Need gradient with respect to mu_diff
                grad_mu_diff = mat_inv_y.mul(2 * grad_output_value)

            if self.needs_input_grad[1]:
                # Compute gradient with respect to the Cholesky factor L
                grad_cholesky_factor = 2 * LinearCG().solve(mm_closure_factory(*covar2_args), chol_covar1)
                grad_cholesky_factor.mul_(grad_output_value)

            if any(self.needs_input_grad[2:]):
                # Compute gradient with respect to covar2
                quad_part = derivative_quadratic_form_factory(*covar2_args)(mat_inv_y, mat_inv_y)

                sample_matrix = torch.sign(torch.randn(self.num_samples, len(mu_diff)))

                for i in range(len(covar2_args)):
                    grad_covar2_args[i] = torch.zeros(covar2_args[i].size())

                def deriv_quad_form_closure(z):
                    I_minus_Tinv_M_z = z - LinearCG().solve(covar2_mv_closure, chol_covar1.t().mv(chol_covar1.mv(z)))
                    Tinv_z = LinearCG().solve(covar2_mv_closure, z)
                    return derivative_quadratic_form_factory(*covar2_args)(Tinv_z, I_minus_Tinv_M_z)

                for z in sample_matrix:
                    for i in range(len(covar2_args)):
                        grad_covar2_args[i].add_(deriv_quad_form_closure(z)[i])

                for i in range(len(covar2_args)):
                    grad_covar2_args[i].div_(self.num_samples)
                    grad_covar2_args[i].add_(-quad_part[i])
                    grad_covar2_args[i].mul_(grad_output_value)

            return tuple([grad_mu_diff] + [grad_cholesky_factor] + grad_covar2_args)

    return TraceLogDetQuadForm


def exact_gp_mll_factory(mm_closure_factory=_default_mm_closure_factor,
                         exact_gp_mml_grad_closure_factory=_default_exact_gp_mml_grad_closure_factory):
    class ExactGPMLL(Function):
        def __init__(self, num_samples=10):
            self.num_samples = num_samples

        def forward(self, *args):
            closure_args = args[:-1]
            labels = args[-1]

            mm_closure = mm_closure_factory(*closure_args)
            mat_inv_labels = LinearCG().solve(mm_closure, labels.unsqueeze(1)).view(-1)
            # Inverse quad form
            res = mat_inv_labels.dot(labels)
            # Log determinant
            logdet, tr_inv = StochasticLQ(num_random_probes=10).evaluate(mm_closure, len(labels),
                                                                         [lambda x: x.log(), lambda x: x.pow(-1)])

            res += logdet
            res += math.log(2 * math.pi) * len(labels)
            res *= -0.5

            self.mat_inv_labels = mat_inv_labels
            self.tr_inv = tr_inv
            self.mm_closure = mm_closure
            self.save_for_backward(*args)
            return labels.new().resize_(1).fill_(res)

        def backward(self, grad_output):
            if exact_gp_mml_grad_closure_factory is None:
                raise NotImplementedError

            closure_args = self.saved_tensors[:-1]
            labels = self.saved_tensors[-1]
            mat_inv_labels = self.mat_inv_labels
            grad_output_value = grad_output.squeeze()[0]

            mm_closure = self.mm_closure
            tr_inv = self.tr_inv
            closure_arg_grads = [None] * len(closure_args)
            labels_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                grad_closure = exact_gp_mml_grad_closure_factory(*(list(closure_args)))
                closure_arg_grads = list(grad_closure(mm_closure, tr_inv, mat_inv_labels, labels, self.num_samples))
                for i, closure_arg_grad in enumerate(closure_arg_grads):
                    if closure_arg_grad is not None:
                        closure_arg_grad.mul_(0.5 * grad_output_value)

            # input_2 gradient
            if self.needs_input_grad[-1]:
                # Need gradient with respect to labels
                labels_grad = mat_inv_labels.mul_(-grad_output_value)

            return tuple(closure_arg_grads + [labels_grad])

    return ExactGPMLL
