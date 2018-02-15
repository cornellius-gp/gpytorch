import math
import torch
from torch.autograd import Function, Variable
from .lincg import LinearCG
from .stochastic_lq import StochasticLQ
from .lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from .. import settings


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
        res = left_factor.transpose(-1, -2).matmul(right_factor).squeeze_()
        return res,
    return closure


def inv_matmul_factory(matmul_closure_factory=_default_matmul_closure_factory,
                       derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class InvMatmul(Function):
        def __init__(self, *args):
            self.args = args

        def forward(self, *args):
            closure_args = self.args + args[:-1]
            matmul_closure = matmul_closure_factory(*closure_args)
            rhs = args[-1]
            res = LinearCG().solve(matmul_closure, rhs)

            self.matmul_closure = matmul_closure
            self.save_for_backward(*(list(args) + [res]))
            return res

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError
            args = self.saved_tensors[:-2]
            res = self.saved_tensors[-1]
            matmul_closure = self.matmul_closure

            arg_grads = [None] * len(args)
            rhs_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                lhs_matrix_grad = LinearCG().solve(matmul_closure, grad_output)
                lhs_matrix_grad = lhs_matrix_grad.mul_(-1)
                if res.ndimension() == 1:
                    res = res.unsqueeze(1)
                if lhs_matrix_grad.ndimension() == 1:
                    lhs_matrix_grad = lhs_matrix_grad.unsqueeze(1)

                arg_grads = list(derivative_quadratic_form_factory(*args)(lhs_matrix_grad.transpose(-1, -2),
                                                                          res.transpose(-1, -2)))

            # input_2 gradient
            if self.needs_input_grad[-1]:
                rhs_grad = LinearCG().solve(matmul_closure, grad_output)

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
            matmul_closure = matmul_closure_factory(*closure_args)
            res = matmul_closure(rhs)

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
            num_random_probes = settings.num_trace_samples.value()
            matrix_size = mu_diff.size(-1)
            self.probe_vectors = mu_diff.new(matrix_size, num_random_probes).bernoulli_().mul_(2).add_(-1)
            self.probe_vector_norms = torch.norm(self.probe_vectors, 2, dim=0)
            probe_vectors = self.probe_vectors.div(self.probe_vector_norms.expand_as(self.probe_vectors))

            probe_vectors.unsqueeze(0)
            if mu_diff.dim() == 2:
                probe_vectors = probe_vectors.expand(mu_diff.size(0), matrix_size, num_random_probes)
                batch_mode = True
            else:
                batch_mode = False

            q_mat, t_mat = lanczos_tridiag(covar2_matmul_closure,
                                           max_iter=settings.max_lanczos_quadrature_iterations.value(),
                                           init_vecs=probe_vectors)
            if not batch_mode:
                t_mat = t_mat.unsqueeze(1)
                q_mat = q_mat.unsqueeze(1)

            eigenvalues, eigenvectors = lanczos_tridiag_to_diag(t_mat)

            slq = StochasticLQ()
            log_det_covar2, = slq.evaluate(q_mat, t_mat, eigenvalues, eigenvectors, [lambda x: x.log()])

            # Tr(K2^{-1}K1)
            covar2_inv_chol_covar1 = LinearCG().solve(covar2_matmul_closure, chol_covar1.transpose(-1, -2))
            trace = (covar2_inv_chol_covar1 * chol_covar1.transpose(-1, -2)).sum(-2).sum(-1)

            # Inverse quad form
            mat_inv_y = LinearCG().solve(covar2_matmul_closure, mu_diff.unsqueeze(-1)).squeeze(-1)
            inv_quad_form = mat_inv_y.mul(mu_diff).sum(-1)

            res = log_det_covar2 + trace + inv_quad_form
            if res.numel() > 1:
                res = res.sum(-1)

            self.save_for_backward(*([mu_diff] + [chol_covar1] + list(covar2_args)))
            self.t_mat_eigenvalues = eigenvalues
            self.t_mat_eigenvectors = eigenvectors
            self.q_mat = q_mat
            self.t_mat = t_mat
            self.covar2_matmul_closure = covar2_matmul_closure
            self.covar2_inv_chol_covar1 = covar2_inv_chol_covar1
            self.mat_inv_y = mat_inv_y

            return res

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError
            grad_output_value = grad_output.squeeze()[0]

            args = self.saved_tensors

            covar2_inv_chol_covar1 = self.covar2_inv_chol_covar1
            covar2_args = args[2:]

            mat_inv_y = self.mat_inv_y.unsqueeze(-2)

            grad_mu_diff = None
            grad_cholesky_factor = None
            grad_covar2_args = [None] * len(covar2_args)

            if self.needs_input_grad[0]:
                # Need gradient with respect to mu_diff
                grad_mu_diff = mat_inv_y.mul(2 * grad_output_value).squeeze(-2)

            if self.needs_input_grad[1]:
                # Compute gradient with respect to the Cholesky factor R
                grad_cholesky_factor = 2 * covar2_inv_chol_covar1.transpose(-1, -2)
                grad_cholesky_factor = grad_cholesky_factor.contiguous()
                grad_cholesky_factor.mul_(grad_output_value)

            if any(self.needs_input_grad[2:]):
                # Compute gradient with respect to covar2
                for i in range(len(covar2_args)):
                    if self.needs_input_grad[i + 2]:
                        grad_covar2_args[i] = covar2_args[i].new().resize_as_(covar2_args[i]).zero_()

                grad_covar2_fn = derivative_quadratic_form_factory(*covar2_args)

                # Get dK/dargs \mu^{\top}K^{-1}\mu^{\top}
                quad_part = grad_covar2_fn(mat_inv_y, mat_inv_y)

                # Get dK/dargs Tr(K^{-1}S)
                left_vectors_trace = covar2_inv_chol_covar1.transpose(-1, -2)
                right_vectors_trace = covar2_inv_chol_covar1.transpose(-1, -2)
                trace_part = grad_covar2_fn(left_vectors_trace, right_vectors_trace)

                # Get dK/dargs log |K|
                num_probes, batch_size, matrix_size, _ = self.t_mat_eigenvectors.size()
                e1_norm_z = self.t_mat_eigenvectors.new(num_probes, batch_size, matrix_size, 1).zero_()
                probe_norms = self.probe_vector_norms.unsqueeze(1).unsqueeze(1).expand(num_probes, batch_size, 1)
                e1_norm_z[:, :, 0, :] = probe_norms

                eigenvecs_times_probes = self.t_mat_eigenvectors.transpose(-2, -1).matmul(e1_norm_z)
                evp_times_inverse_eigenvals = (1 / self.t_mat_eigenvalues.unsqueeze(-1)) * eigenvecs_times_probes
                t_mat_inverse_times_probes = self.t_mat_eigenvectors.matmul(evp_times_inverse_eigenvals)
                covar_inverse_times_probes = self.q_mat.matmul(t_mat_inverse_times_probes)
                coef = math.sqrt(1. / self.probe_vectors.size(-1))
                if batch_size > 1:
                    left_vectors_logdet = covar_inverse_times_probes.squeeze().mul(coef).permute(1, 0, 2)
                    right_vectors_logdet = self.probe_vectors.mul(coef).t().unsqueeze(0).expand_as(left_vectors_logdet)
                else:
                    left_vectors_logdet = covar_inverse_times_probes.squeeze().mul(coef)
                    right_vectors_logdet = self.probe_vectors.mul(coef).t()

                grad_covar2_args = list(grad_covar2_fn(left_vectors_logdet, right_vectors_logdet))

                for i in range(len(covar2_args)):
                    if grad_covar2_args[i] is not None:
                        grad_covar2_args[i].add_(-trace_part[i])
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
            num_random_probes = settings.num_trace_samples.value()
            matrix_size = labels.size(-2)
            self.probe_vectors = labels.new(matrix_size, num_random_probes).bernoulli_().mul_(2).add_(-1)
            self.probe_vector_norms = torch.norm(self.probe_vectors, 2, dim=0)

            probe_vectors = self.probe_vectors.div(self.probe_vector_norms.expand_as(self.probe_vectors))
            probe_vectors.unsqueeze(0)
            if labels.dim() == 3:
                probe_vectors = probe_vectors.expand(labels.size(0), matrix_size, num_random_probes)
                batch_mode = True
            else:
                batch_mode = False

            q_mat, t_mat = lanczos_tridiag(matmul_closure,
                                           max_iter=settings.max_lanczos_quadrature_iterations.value(),
                                           init_vecs=probe_vectors)

            if not batch_mode:
                t_mat = t_mat.unsqueeze(1)
                q_mat = q_mat.unsqueeze(1)

            eigenvalues, eigenvectors = lanczos_tridiag_to_diag(t_mat)

            slq = StochasticLQ()
            logdet, = slq.evaluate(q_mat, t_mat, eigenvalues, eigenvectors, [lambda x: x.log()])

            res += logdet
            res += math.log(2 * math.pi) * labels.size(-2)
            res *= -0.5

            self.t_mat_eigenvalues = eigenvalues
            self.t_mat_eigenvectors = eigenvectors
            self.q_mat = q_mat
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
            mat_inv_labels = self.mat_inv_labels

            closure_arg_grads = [None] * len(closure_args)
            labels_grad = None

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                for i in range(len(closure_args)):
                    if self.needs_input_grad[i]:
                        closure_arg_grads[i] = closure_args[i].new().resize_as_(closure_args[i]).zero_()
                    else:
                        closure_arg_grads[i] = None

                function = derivative_quadratic_form_factory(*closure_args)
                quad_form_part = function(mat_inv_labels.transpose(-1, -2), mat_inv_labels.transpose(-1, -2))

                # probe_vectors is (batch_size x matrix_size x num_probes)
                # t_mat_eigenvectors is (num_probes x batch_size x matrix_size x matrix_size)
                num_probes, batch_size, matrix_size, _ = self.t_mat_eigenvectors.size()
                e1_norm_z = self.t_mat_eigenvectors.new(num_probes, batch_size, matrix_size, 1).zero_()
                e1_norm_z[:, :, 0, :] = self.probe_vector_norms
                eigenvecs_times_probes = self.t_mat_eigenvectors.transpose(-2, -1).matmul(e1_norm_z)
                evp_times_inverse_eigenvals = (1 / self.t_mat_eigenvalues.unsqueeze(-1)) * eigenvecs_times_probes
                t_mat_inverse_times_probes = self.t_mat_eigenvectors.matmul(evp_times_inverse_eigenvals)
                covar_inverse_times_probes = self.q_mat.matmul(t_mat_inverse_times_probes)
                coef = math.sqrt(1. / self.probe_vectors.size(-1))
                left_vectors = covar_inverse_times_probes.squeeze().mul(coef)
                right_vectors = self.probe_vectors.mul(coef)
                closure_arg_grads = list(function(left_vectors, right_vectors.transpose(-1, -2)))

                for i in range(len(closure_args)):
                    if self.needs_input_grad[i] and closure_arg_grads[i] is not None:
                        # This check makes sure we're only doing math if we need to (i.e. grads aren't zero)
                        if closure_arg_grads[i].sum():
                            closure_arg_grads[i] = quad_form_part[i].add_(-closure_arg_grads[i])
                            if closure_arg_grads[i].ndimension() == 3:
                                closure_arg_grads[i].mul_(0.5 * grad_output.unsqueeze(1).unsqueeze(2))
                            elif closure_arg_grads[i].ndimension() == 2:
                                closure_arg_grads[i].mul_(0.5 * grad_output.unsqueeze(1))
                            elif closure_arg_grads[i].numel() > 1:
                                closure_arg_grads[i].mul_(0.5 * grad_output)
                            else:
                                closure_arg_grads[i].mul_(0.5 * grad_output.sum())

            # input_2 gradient
            if self.needs_input_grad[-1]:
                # Need gradient with respect to labels
                if mat_inv_labels.ndimension() == 3:
                    labels_grad = mat_inv_labels.squeeze(-1).mul_(-grad_output.unsqueeze(1))
                else:
                    labels_grad = mat_inv_labels.squeeze(-1).mul_(-grad_output)

            res = tuple(closure_arg_grads + [labels_grad])
            return res

    return ExactGPMLL


def root_decomposition_factory(matmul_closure_factory=_default_matmul_closure_factory,
                               derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class RootDecomposition(Function):
        def __init__(self, cls, size, max_iter, batch_size=None, root=True, inverse=False, initial_vector=None):
            self.cls = cls
            self.size = size
            self.max_iter = max_iter
            self.batch_size = batch_size
            self.root = root
            self.inverse = inverse
            self.initial_vector = initial_vector.data if isinstance(initial_vector, Variable) else initial_vector

        def forward(self, *args):
            matmul_closure = matmul_closure_factory(*args)

            def tensor_matmul_closure(rhs):
                return matmul_closure(rhs)

            # Do lanczos
            q_mat, t_mat = lanczos_tridiag(tensor_matmul_closure, self.max_iter,
                                           tensor_cls=self.cls, batch_size=self.batch_size,
                                           n_dims=self.size, init_vecs=self.initial_vector)
            if self.batch_size is None:
                q_mat = q_mat.unsqueeze(-3)
                t_mat = t_mat.unsqueeze(-3)
            if t_mat.ndimension() == 3:  # If we only used one probe vector
                q_mat = q_mat.unsqueeze(0)
                t_mat = t_mat.unsqueeze(0)
            n_probes = t_mat.size(0)

            eigenvalues, eigenvectors = lanczos_tridiag_to_diag(t_mat)

            # Get orthogonal matrix and eigenvalue roots
            q_mat = q_mat.matmul(eigenvectors)
            root_evals = eigenvalues.sqrt()

            # Store q_mat * t_mat_chol
            # Decide if we're computing the inverse, or the regular root
            root = q_mat.new()
            inverse = q_mat.new()
            if self.inverse:
                inverse = q_mat / root_evals.unsqueeze(-2)
                self.__root_inverse = inverse
            if self.root:
                root = q_mat * root_evals.unsqueeze(-2)

            self.__q_mat = q_mat
            self.__root_evals = root_evals
            self.save_for_backward(*args)

            if self.batch_size is None:
                root = root.squeeze(1) if root.numel() else root
                inverse = inverse.squeeze(1) if inverse.numel() else inverse
            if n_probes == 1:
                root = root.squeeze(0) if root.numel() else root
                inverse = inverse.squeeze(0) if inverse.numel() else inverse
            return root, inverse

        def backward(self, root_grad_output, inverse_grad_output):
            # Taken from http://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf
            if any(self.needs_input_grad):
                args = self.saved_tensors
                if root_grad_output.numel() == 1 and root_grad_output[0] == 0:
                    root_grad_output = None
                if inverse_grad_output.numel() == 1 and inverse_grad_output[0] == 0:
                    inverse_grad_output = None

                if root_grad_output is not None:
                    if root_grad_output.ndimension() == 2:
                        root_grad_output = root_grad_output.unsqueeze(0)
                    if root_grad_output.ndimension() == 3:
                        root_grad_output = root_grad_output.unsqueeze(0)
                if inverse_grad_output is not None:
                    if inverse_grad_output.ndimension() == 2:
                        inverse_grad_output = inverse_grad_output.unsqueeze(0)
                    if inverse_grad_output.ndimension() == 3:
                        inverse_grad_output = inverse_grad_output.unsqueeze(0)

                # Get root inverse
                if self.inverse:
                    root_inverse_t = self.__root_inverse.transpose(-1, -2)
                else:
                    q_mat = self.__q_mat
                    root_evals = self.__root_evals
                    q_mat_t = q_mat.transpose(-1, -2)
                    root_inverse_t = q_mat_t / root_evals.unsqueeze(-1)

                # Left factor:
                left_factor = root_inverse_t.new(root_inverse_t.size(0), root_inverse_t.size(1),
                                                 root_inverse_t.size(-2), root_inverse_t.size(-1)).zero_()
                if root_grad_output is not None:
                    left_factor.add_(root_grad_output.transpose(-1, -2))
                if inverse_grad_output is not None:
                    # -root^-T grad_output.T root^-T
                    left_factor.sub_(torch.matmul(root_inverse_t, inverse_grad_output).matmul(root_inverse_t))

                right_factor = root_inverse_t.div(2.)

                # Fix batches
                left_factor = left_factor.permute(1, 0, 2, 3).contiguous()
                left_factor = left_factor.view(root_inverse_t.size(1), -1, left_factor.size(-1))
                right_factor = right_factor.permute(1, 0, 2, 3).contiguous()
                right_factor = right_factor.view(root_inverse_t.size(1), -1, right_factor.size(-1))
                res = derivative_quadratic_form_factory(*args)(left_factor, right_factor)
                return tuple(res)
            else:
                pass

    return RootDecomposition
