from torch.autograd import Function
from .lincg import LinearCG
from .lanczos_quadrature import StochasticLQ
import torch
import math
import gpytorch.functions


def _default_matmul_closure_factor(mat):
    return mat


def _default_derivative_quadratic_form_factory(mat):
    def closure(left_vectors, right_vectors):
        if left_vectors.ndimension() == 1:
            left_factor = left_vectors.unsqueeze(0)
            right_factor = right_vectors.unsqueeze(0)
        else:
            left_factor = left_vectors
            right_factor = right_vectors
        left_factor.unsqueeze_(2)
        right_factor.unsqueeze_(1)
        res = (left_factor * right_factor).sum(dim=0).squeeze_()
        return res,
    return closure


def inv_matmul_factory(matmul_closure_factory=_default_matmul_closure_factor,
                       derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class InvMatmul(Function):
        def __init__(self, *args):
            self.args = args

        def forward(self, *args):
            closure_args = self.args + args[:-1]
            rhs = args[-1]
            res = LinearCG().solve(matmul_closure_factory(*closure_args), rhs)
            self.save_for_backward(*(list(args) + [res]))
            return res

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError
            args = self.saved_tensors[:-2]
            closure_args = self.args + args
            res = self.saved_tensors[-1]

            arg_grads = [None] * len(args)
            rhs_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                lhs_matrix_grad = LinearCG().solve(matmul_closure_factory(*closure_args), grad_output)
                lhs_matrix_grad = lhs_matrix_grad.mul_(-1)
                if res.ndimension() == 1:
                    res = res.unsqueeze(1)
                if lhs_matrix_grad.ndimension() == 1:
                    lhs_matrix_grad = lhs_matrix_grad.unsqueeze(1)

                arg_grads = list(derivative_quadratic_form_factory(*args)(lhs_matrix_grad.t(), res.t()))

            # input_2 gradient
            if self.needs_input_grad[-1]:
                rhs_grad = LinearCG().solve(matmul_closure_factory(*closure_args), grad_output)

            return tuple(arg_grads + [rhs_grad])

    return InvMatmul


def matmul_factory(matmul_closure_factory=_default_matmul_closure_factor,
                   derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class Matmul(Function):
        def __init__(self, *args):
            self.args = args

        def forward(self, *args):
            closure_args = self.args + args[:-1]
            rhs = args[-1]
            res = matmul_closure_factory(*closure_args)(rhs)
            self.save_for_backward(*args)
            return res

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError
            args = self.saved_tensors[:-1]
            rhs = self.saved_tensors[-1]
            closure_args = self.args + args

            arg_grads = [None] * len(args)
            rhs_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                if rhs.ndimension() == 1:
                    rhs = rhs.unsqueeze(1)
                if grad_output.ndimension() == 1:
                    grad_output_matrix = grad_output.unsqueeze(1)
                else:
                    grad_output_matrix = grad_output

                arg_grads = list(derivative_quadratic_form_factory(*args)(grad_output_matrix.t(), rhs.t()))

            # input_2 gradient
            if self.needs_input_grad[-1]:
                rhs_grad = matmul_closure_factory(*closure_args)(grad_output)

            return tuple(arg_grads + [rhs_grad])

    return Matmul


def trace_logdet_quad_form_factory(matmul_closure_factory=_default_matmul_closure_factor,
                                   derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class TraceLogDetQuadForm(Function):
        def forward(self, mu_diff, chol_covar1, *covar2_args):
            covar2_matmul_closure = matmul_closure_factory(*covar2_args)

            # log |K2|
            log_det_covar2, = StochasticLQ(num_random_probes=10).evaluate(covar2_matmul_closure,
                                                                          len(mu_diff),
                                                                          [lambda x: x.log()])

            # Tr(K2^{-1}K1)
            if gpytorch.functions.fastest:
                sample_matrix = torch.sign(torch.randn(len(mu_diff), gpytorch.functions.num_trace_samples))
                rhs_vectors = chol_covar1.t().mm(chol_covar1.mm(sample_matrix))
                mat_inv_vectors = LinearCG().solve(covar2_matmul_closure, rhs_vectors)
                trace = (mat_inv_vectors * sample_matrix).sum() / gpytorch.functions.num_trace_samples
            else:
                cov2_inv_cov1 = LinearCG().solve(covar2_matmul_closure, chol_covar1.t().matmul(chol_covar1))
                self.cov2_inv_cov1 = cov2_inv_cov1
                trace = cov2_inv_cov1.trace()

            # Inverse quad form
            mat_inv_y = LinearCG().solve(covar2_matmul_closure, mu_diff)
            inv_quad_form = mat_inv_y.dot(mu_diff)

            res = log_det_covar2 + trace + inv_quad_form

            self.save_for_backward(*([mu_diff] + [chol_covar1] + list(covar2_args)))
            self.covar2_matmul_closure = covar2_matmul_closure
            self.mat_inv_y = mat_inv_y

            return mu_diff.new().resize_(1).fill_(res)

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError
            grad_output_value = grad_output.squeeze()[0]

            args = self.saved_tensors

            mu_diff = args[0]
            chol_covar1 = args[1]
            covar2_args = args[2:]

            mat_inv_y = self.mat_inv_y
            covar2_matmul_closure = self.covar2_matmul_closure

            grad_mu_diff = None
            grad_cholesky_factor = None
            grad_covar2_args = [None] * len(covar2_args)

            if self.needs_input_grad[0]:
                # Need gradient with respect to mu_diff
                grad_mu_diff = mat_inv_y.mul(2 * grad_output_value)

            if self.needs_input_grad[1]:
                # Compute gradient with respect to the Cholesky factor L
                grad_cholesky_factor = 2 * LinearCG().solve(matmul_closure_factory(*covar2_args), chol_covar1)
                grad_cholesky_factor.mul_(grad_output_value)

            if any(self.needs_input_grad[2:]):
                # Compute gradient with respect to covar2
                for i in range(len(covar2_args)):
                    if self.needs_input_grad[i + 2]:
                        grad_covar2_args[i] = torch.zeros(covar2_args[i].size())

                if gpytorch.functions.fastest:
                    quad_part = derivative_quadratic_form_factory(*covar2_args)(mat_inv_y, mat_inv_y)

                    sample_matrix = torch.sign(torch.randn(len(mu_diff), gpytorch.functions.num_trace_samples))
                    rhs_vectors = chol_covar1.t().mm(chol_covar1.mm(sample_matrix))
                    I_minus_Tinv_M_vectors = sample_matrix - LinearCG().solve(covar2_matmul_closure, rhs_vectors)
                    Tinv_vectors = LinearCG().solve(covar2_matmul_closure, sample_matrix)
                    grad_covar2_args = list(derivative_quadratic_form_factory(*covar2_args)(Tinv_vectors.t(),
                                                                                            I_minus_Tinv_M_vectors.t()))

                    for i in range(len(covar2_args)):
                        if grad_covar2_args[i] is not None:
                            grad_covar2_args[i].div_(gpytorch.functions.num_trace_samples)
                            grad_covar2_args[i].add_(-quad_part[i])
                            grad_covar2_args[i].mul_(grad_output_value)
                else:
                    left_vectors_1 = torch.eye(len(mu_diff)) - torch.ger(mu_diff, mat_inv_y)
                    left_vectors_1 = LinearCG().solve(covar2_matmul_closure, left_vectors_1)
                    right_vectors_1_t = torch.eye(len(mu_diff)) - self.cov2_inv_cov1
                    right_vectors_1 = right_vectors_1_t.t()

                    left_vectors_2 = torch.ger(mat_inv_y, mat_inv_y)
                    right_vectors_2 = self.cov2_inv_cov1.t()

                    quad_derivative_1 = list(derivative_quadratic_form_factory(*covar2_args)(left_vectors_1,
                                                                                             right_vectors_1))
                    quad_derivative_2 = list(derivative_quadratic_form_factory(*covar2_args)(left_vectors_2,
                                                                                             right_vectors_2))

                    for i in range(len(covar2_args)):
                        if grad_covar2_args[i] is not None:
                            grad_covar2_args[i].add_(quad_derivative_1[i])
                            grad_covar2_args[i].add_(-quad_derivative_2[i])
                            grad_covar2_args[i].mul_(grad_output_value)

            return tuple([grad_mu_diff] + [grad_cholesky_factor] + grad_covar2_args)

    return TraceLogDetQuadForm


def exact_gp_mll_factory(matmul_closure_factory=_default_matmul_closure_factor,
                         derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class ExactGPMLL(Function):
        def forward(self, *args):
            closure_args = args[:-1]
            labels = args[-1]

            matmul_closure = matmul_closure_factory(*closure_args)
            mat_inv_labels = LinearCG().solve(matmul_closure, labels)
            # Inverse quad form
            res = mat_inv_labels.dot(labels)
            # Log determinant
            logdet, = StochasticLQ(num_random_probes=10).evaluate(matmul_closure, len(labels), [lambda x: x.log()])

            res += logdet
            res += math.log(2 * math.pi) * len(labels)
            res *= -0.5

            self.mat_inv_labels = mat_inv_labels
            self.matmul_closure = matmul_closure
            self.save_for_backward(*args)
            return labels.new().resize_(1).fill_(res)

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError

            closure_args = self.saved_tensors[:-1]
            labels = self.saved_tensors[-1]
            mat_inv_labels = self.mat_inv_labels
            grad_output_value = grad_output.squeeze()[0]

            matmul_closure = self.matmul_closure
            closure_arg_grads = [None] * len(closure_args)
            labels_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                for i in range(len(closure_args)):
                    if self.needs_input_grad[i]:
                        closure_arg_grads[i] = torch.zeros(closure_args[i].size())
                    else:
                        closure_arg_grads[i] = None

                if gpytorch.functions.fastest:
                    num_samples = gpytorch.functions.num_trace_samples
                    quad_form_part = derivative_quadratic_form_factory(*closure_args)(mat_inv_labels, mat_inv_labels)

                    sample_matrix = torch.sign(torch.randn(len(labels), num_samples))
                    left_vectors = LinearCG().solve(matmul_closure, sample_matrix).t()
                    right_vectors = sample_matrix.t()

                    closure_arg_grads = list(derivative_quadratic_form_factory(*closure_args)(left_vectors,
                                                                                              right_vectors))

                    for i in range(len(closure_args)):
                        if self.needs_input_grad[i]:
                            closure_arg_grads[i] = quad_form_part[i].add_(-closure_arg_grads[i].div_(num_samples))
                            closure_arg_grads[i].mul_(0.5 * grad_output_value)
                else:
                    left_vectors = torch.ger(labels, mat_inv_labels) - torch.eye(len(labels))
                    left_vectors = LinearCG().solve(matmul_closure, left_vectors)
                    right_vectors = torch.eye(len(labels))
                    closure_arg_grads = list(derivative_quadratic_form_factory(*closure_args)(left_vectors,
                                                                                              right_vectors))

                    for i in range(len(closure_args)):
                        if self.needs_input_grad[i]:
                            closure_arg_grads[i].mul_(0.5 * grad_output_value)

            # input_2 gradient
            if self.needs_input_grad[-1]:
                # Need gradient with respect to labels
                labels_grad = mat_inv_labels.mul_(-grad_output_value)

            return tuple(closure_arg_grads + [labels_grad])

    return ExactGPMLL
