from torch.autograd import Function
from .lincg import LinearCG
from .lanczos_quadrature import StochasticLQ
from .trace import trace_components
import torch
import math


def _default_matmul_closure_factory(mat):
    return mat


def _default_t_matmul_closure_factory(mat):
    return mat.transpose(-1, -2).matmul


def _default_derivative_quadratic_form_factory(mat):
    def closure(left_vectors, right_vectors):
        if left_vectors.ndimension() == 1:
            left_factor = left_vectors.unsqueeze(0).contiguous()
            right_factor = right_vectors.unsqueeze(0).contiguous()
        else:
            left_factor = left_vectors.contiguous()
            right_factor = right_vectors.contiguous()
        left_factor = left_factor.unsqueeze(-1)
        right_factor = right_factor.unsqueeze(-2)
        res = (left_factor * right_factor).sum(dim=-3).squeeze_()
        return res,
    return closure


def inv_matmul_factory(matmul_closure_factory=_default_matmul_closure_factory,
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


def matmul_factory(matmul_closure_factory=_default_matmul_closure_factory,
                   derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory,
                   t_matmul_closure_factory=_default_t_matmul_closure_factory):
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

                if grad_output_matrix.ndimension() == 3:
                    arg_grads = list(derivative_quadratic_form_factory(*args)(grad_output_matrix.transpose(1, 2),
                                                                              rhs.transpose(1, 2)))
                else:
                    arg_grads = list(derivative_quadratic_form_factory(*args)(grad_output_matrix.t(), rhs.t()))

            # input_2 gradient
            if self.needs_input_grad[-1]:
                rhs_grad = t_matmul_closure_factory(*closure_args)(grad_output)

            return tuple(arg_grads + [rhs_grad])

    return Matmul


def trace_logdet_quad_form_factory(matmul_closure_factory=_default_matmul_closure_factory,
                                   derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class TraceLogDetQuadForm(Function):
        def forward(self, mu_diff, chol_covar1, *covar2_args):
            covar2_matmul_closure = matmul_closure_factory(*covar2_args)

            # log |K2|
            slq = StochasticLQ(num_random_probes=10, cls=type(covar2_args[0]))
            if mu_diff.ndimension() == 2:
                log_det_covar2, = slq.evaluate(covar2_matmul_closure, mu_diff.size(-1),
                                               [lambda x: x.log()], batch_size=mu_diff.size(0))
            else:
                log_det_covar2, = slq.evaluate(covar2_matmul_closure, mu_diff.size(-1), [lambda x: x.log()])

            # Tr(K2^{-1}K1)
            def matmul_closure(sample_matrix):
                if chol_covar1.ndimension() == 3:
                    sample_matrix = sample_matrix.unsqueeze(0)
                    sample_matrix = sample_matrix.expand(chol_covar1.size(0), sample_matrix.size(1),
                                                         sample_matrix.size(2))
                rhs_vectors = chol_covar1.transpose(-1, -2).contiguous().matmul(chol_covar1.matmul(sample_matrix))
                return LinearCG().solve(covar2_matmul_closure, rhs_vectors)

            sample_matrix, mat_inv_vectors = trace_components(None, matmul_closure, size=mu_diff.size(-1),
                                                              tensor_cls=type(chol_covar1))
            trace = (sample_matrix * mat_inv_vectors).sum(-2).sum(-1)

            # Inverse quad form
            mat_inv_y = LinearCG().solve(covar2_matmul_closure, mu_diff.unsqueeze(-1)).squeeze(-1)
            inv_quad_form = mat_inv_y.mul(mu_diff).sum(-1)

            res = log_det_covar2 + trace + inv_quad_form
            if res.numel() > 1:
                res = res.sum(-1)

            self.save_for_backward(*([mu_diff] + [chol_covar1] + list(covar2_args)))
            self.covar2_matmul_closure = covar2_matmul_closure
            self.mat_inv_y = mat_inv_y

            return res

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError
            grad_output_value = grad_output.squeeze()[0]

            args = self.saved_tensors

            mu_diff = args[0]
            chol_covar1 = args[1]
            covar2_args = args[2:]

            mat_inv_y = self.mat_inv_y.unsqueeze(-2)
            covar2_matmul_closure = self.covar2_matmul_closure

            grad_mu_diff = None
            grad_cholesky_factor = None
            grad_covar2_args = [None] * len(covar2_args)

            if self.needs_input_grad[0]:
                # Need gradient with respect to mu_diff
                grad_mu_diff = mat_inv_y.mul(2 * grad_output_value).squeeze(-2)

            if self.needs_input_grad[1]:
                # Compute gradient with respect to the Cholesky factor L
                grad_cholesky_factor = 2 * LinearCG().solve(matmul_closure_factory(*covar2_args),
                                                            chol_covar1.transpose(-1, -2)).transpose(-1, -2)
                grad_cholesky_factor = grad_cholesky_factor.contiguous()
                grad_cholesky_factor.mul_(grad_output_value)

            if any(self.needs_input_grad[2:]):
                # Compute gradient with respect to covar2
                for i in range(len(covar2_args)):
                    if self.needs_input_grad[i + 2]:
                        grad_covar2_args[i] = covar2_args[i].new().resize_as_(covar2_args[i]).zero_()

                quad_part = derivative_quadratic_form_factory(*covar2_args)(mat_inv_y, mat_inv_y)

                def right_matmul_closure(sample_matrix):
                    rhs_vectors = chol_covar1.transpose(-1, -2).contiguous().matmul(chol_covar1.matmul(sample_matrix))
                    return sample_matrix - LinearCG().solve(covar2_matmul_closure, rhs_vectors)

                def left_matmul_closure(sample_matrix):
                    if chol_covar1.ndimension() == 3:
                        sample_matrix = sample_matrix.unsqueeze(0)
                        sample_matrix = sample_matrix.expand(chol_covar1.size(0), sample_matrix.size(1),
                                                             sample_matrix.size(2))
                    return LinearCG().solve(covar2_matmul_closure, sample_matrix)

                left_vectors, right_vectors = trace_components(left_matmul_closure, right_matmul_closure,
                                                               size=mu_diff.size(-1), tensor_cls=type(mat_inv_y))

                grad_covar2_fn = derivative_quadratic_form_factory(*covar2_args)
                grad_covar2_args = list(grad_covar2_fn(left_vectors.transpose(-1, -2),
                                                       right_vectors.transpose(-1, -2)))

                for i in range(len(covar2_args)):
                    if grad_covar2_args[i] is not None:
                        grad_covar2_args[i].add_(-quad_part[i])
                        grad_covar2_args[i].mul_(grad_output_value)

            res = tuple([grad_mu_diff] + [grad_cholesky_factor] + grad_covar2_args)
            return res

    return TraceLogDetQuadForm


def exact_gp_mll_factory(matmul_closure_factory=_default_matmul_closure_factory,
                         derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class ExactGPMLL(Function):
        def forward(self, *args):
            closure_args = args[:-1]
            labels = args[-1]
            labels = labels.unsqueeze(-1)

            matmul_closure = matmul_closure_factory(*closure_args)
            mat_inv_labels = LinearCG().solve(matmul_closure, labels)
            # Inverse quad form
            res = (mat_inv_labels * labels).sum(-1).sum(-1)
            # Log determinant
            slq = StochasticLQ(num_random_probes=10, cls=type(closure_args[0]))
            if labels.ndimension() == 3:
                logdet, = slq.evaluate(matmul_closure, labels.size(1), [lambda x: x.log()], batch_size=labels.size(0))
            else:
                logdet, = slq.evaluate(matmul_closure, labels.size(0), [lambda x: x.log()])

            res += logdet
            res += math.log(2 * math.pi) * labels.size(-2)
            res *= -0.5

            self.mat_inv_labels = mat_inv_labels
            self.matmul_closure = matmul_closure
            self.save_for_backward(*args)
            if torch.is_tensor(res):
                return res
            return labels.new().resize_(1).fill_(res)

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError

            closure_args = self.saved_tensors[:-1]
            labels = self.saved_tensors[-1].unsqueeze(-1)
            mat_inv_labels = self.mat_inv_labels
            grad_output_value = grad_output.squeeze()[0]

            matmul_closure = self.matmul_closure
            closure_arg_grads = [None] * len(closure_args)
            labels_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                for i in range(len(closure_args)):
                    if self.needs_input_grad[i]:
                        closure_arg_grads[i] = closure_args[i].new().resize_as_(closure_args[i]).zero_()
                    else:
                        closure_arg_grads[i] = None

                if labels.ndimension() == 3:
                    function = derivative_quadratic_form_factory(*closure_args)
                    quad_form_part = function(mat_inv_labels, mat_inv_labels)
                else:
                    function = derivative_quadratic_form_factory(*closure_args)
                    quad_form_part = function(mat_inv_labels.squeeze(1), mat_inv_labels.squeeze(1))

                def left_matmul_closure(sample_matrix):
                    return LinearCG().solve(matmul_closure, sample_matrix)

                if labels.ndimension() == 3:
                    left_vectors, right_vectors = trace_components(left_matmul_closure, None, size=labels.size(1),
                                                                   tensor_cls=type(mat_inv_labels),
                                                                   dim_num=labels.size(0))
                    function = derivative_quadratic_form_factory(*closure_args)
                    closure_arg_grads = list(function(left_vectors.transpose(1, 2), right_vectors.transpose(1, 2)))
                else:
                    left_vectors, right_vectors = trace_components(left_matmul_closure, None, size=labels.size(0),
                                                                   tensor_cls=type(mat_inv_labels))
                    closure_arg_grads = list(derivative_quadratic_form_factory(*closure_args)(left_vectors.t(),
                                                                                              right_vectors.t()))

                for i in range(len(closure_args)):
                    if self.needs_input_grad[i]:
                        closure_arg_grads[i] = quad_form_part[i].add_(-closure_arg_grads[i])
                        closure_arg_grads[i].mul_(0.5 * grad_output_value)

            # input_2 gradient
            if self.needs_input_grad[-1]:
                # Need gradient with respect to labels
                labels_grad = mat_inv_labels.mul_(-grad_output_value).squeeze(-1)

            return tuple(closure_arg_grads + [labels_grad])

    return ExactGPMLL
