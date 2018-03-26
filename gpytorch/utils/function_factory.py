from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Function, Variable
from .linear_cg import linear_cg
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
            rhs_shape = rhs.shape
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
                # TODO: the matmul closure factory should return the same
                # shape as rhs.
                rhs_grad = rhs_grad.view(rhs_shape)

            return tuple(arg_grads + [rhs_grad])

    return Matmul


def inv_matmul_factory(matmul_closure_factory=_default_matmul_closure_factory,
                       derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class InvMatmul(Function):
        def __init__(self, preconditioner=None):
            self.preconditioner = preconditioner

        def forward(self, *args):
            closure_args = args[:-1]
            rhs = args[-1]
            matmul_closure = matmul_closure_factory(*closure_args)

            self.is_vector = False
            if rhs.ndimension() == 1:
                rhs = rhs.unsqueeze(-1)
                self.is_vector = True

            # Perform solves (for inv_quad) and tridiagonalization (for estimating log_det)
            res = linear_cg(matmul_closure,
                            rhs,
                            max_iter=settings.max_cg_iterations.value(),
                            preconditioner=self.preconditioner)

            if self.is_vector:
                res.squeeze_(-1)

            self.matmul_closure = matmul_closure
            self.save_for_backward(*(list(args) + [res]))
            return res

        def backward(self, grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError

            # Extract items that were saved
            closure_args = self.saved_tensors[:-2]
            rhs = self.saved_tensors[-2]
            rhs_solves = self.saved_tensors[-1]

            # Define gradient placeholders
            arg_grads = [None] * len(closure_args)
            rhs_grad = None
            if not any(self.needs_input_grad):
                return arg_grads, rhs_grad

            # De-vectorize objects
            if self.is_vector:
                rhs = rhs.unsqueeze(-1)
                rhs_solves = rhs_solves.unsqueeze(-1)
                grad_output = grad_output.unsqueeze(-1)

            # Compute self^{-1} grad_output
            grad_output_solves = linear_cg(self.matmul_closure, grad_output,
                                           max_iter=settings.max_cg_iterations.value(),
                                           preconditioner=self.preconditioner)

            # input_1 gradient
            if any(self.needs_input_grad[:-1]):
                derivative_fn = derivative_quadratic_form_factory(*closure_args)
                arg_grads = list(derivative_fn(grad_output_solves.transpose(-1, -2),
                                               rhs_solves.mul(-1).transpose(-1, -2)))

            # input_2 gradient
            if self.needs_input_grad[-1]:
                rhs_grad = grad_output_solves
                if self.is_vector:
                    rhs_grad.squeeze_(-1)

            return tuple(arg_grads + [rhs_grad])

    return InvMatmul


def inv_quad_log_det_factory(matmul_closure_factory=_default_matmul_closure_factory,
                             derivative_quadratic_form_factory=_default_derivative_quadratic_form_factory):
    class InvQuadLogDet(Function):
        """
        Given a PSD matrix A (or a batch of PSD matrices A), this function computes one or both
        of the following
        - The matrix solves A^{-1} b
        - logdet(A)
        """
        def __init__(self, matrix_size=0, batch_size=None, tensor_cls=None,
                     inv_quad=False, log_det=False, preconditioner=None):
            if not matrix_size:
                raise RuntimeError('Matrix size must be set')
            if tensor_cls is None:
                raise RuntimeError('tensor_cls must be set')
            if not (inv_quad or log_det):
                raise RuntimeError('Either inv_quad or log_det must be true (or both)')
            self.matrix_size = matrix_size
            self.batch_size = batch_size
            self.tensor_cls = tensor_cls
            self.inv_quad = inv_quad
            self.log_det = log_det
            self.preconditioner = preconditioner

        def forward(self, *args):
            """
            *args - The arguments representing the PSD matrix A (or batch of PSD matrices A)
            If self.inv_quad is true, the last entry in *args is inv_quad_rhs (Tensor)
            - the RHS of the matrix solves.

            Returns:
            - (Tensor) The solves (or None, if self.inv_quad is False)
            - (Scalar) The log determinant (or None, self.if log_det is False)
            """
            closure_args = None
            inv_quad_rhs = None
            if self.inv_quad:
                closure_args = args[:-1]
                inv_quad_rhs = args[-1]
            else:
                closure_args = args
            matmul_closure = matmul_closure_factory(*closure_args)

            # Collect terms for LinearCG
            # We use LinearCG for both matrix solves and for stochastically estimating the log det
            rhs_list = []
            num_random_probes = 0
            num_inv_quad_solves = 0

            # Probe vector for lanczos quadrature (log_det estimation)
            if self.log_det:
                num_random_probes = settings.num_trace_samples.value()
                self.probe_vectors = self.tensor_cls(self.matrix_size, num_random_probes).bernoulli_().mul_(2).add_(-1)
                self.probe_vector_norms = torch.norm(self.probe_vectors, 2, dim=-2, keepdim=True)
                if self.batch_size is not None:
                    self.probe_vectors = self.probe_vectors.unsqueeze(0).expand(self.batch_size,
                                                                                self.matrix_size,
                                                                                num_random_probes)
                    self.probe_vector_norms = self.probe_vector_norms.unsqueeze(0).expand(self.batch_size, 1,
                                                                                          num_random_probes)
                probe_vectors = self.probe_vectors.div(self.probe_vector_norms)
                rhs_list.append(probe_vectors)

            # RHS for inv_quad
            self.is_vector = False
            if self.inv_quad:
                if inv_quad_rhs.ndimension() == 1:
                    inv_quad_rhs = inv_quad_rhs.unsqueeze(-1)
                    self.is_vector = True
                rhs_list.append(inv_quad_rhs)
                num_inv_quad_solves = inv_quad_rhs.size(-1)

            # Perform solves (for inv_quad) and tridiagonalization (for estimating log_det)
            rhs = torch.cat(rhs_list, -1)
            t_mat = None
            if self.log_det:
                solves, t_mat = linear_cg(matmul_closure, rhs, n_tridiag=num_random_probes,
                                          max_iter=settings.max_lanczos_quadrature_iterations.value(),
                                          preconditioner=self.preconditioner)
            else:
                solves = linear_cg(matmul_closure, rhs, n_tridiag=num_random_probes,
                                   max_iter=settings.max_lanczos_quadrature_iterations.value(),
                                   preconditioner=self.preconditioner)

            # Final values to return
            log_det_term = self.tensor_cls()
            inv_quad_term = self.tensor_cls()

            # Compute log_det from tridiagonalization
            if self.log_det:
                if self.batch_size is None:
                    t_mat = t_mat.unsqueeze(1)
                eigenvalues, eigenvectors = lanczos_tridiag_to_diag(t_mat)
                slq = StochasticLQ()
                matrix_size = rhs.size(-2)
                log_det_term, = slq.evaluate(t_mat, matrix_size, eigenvalues, eigenvectors, [lambda x: x.log()])

            # Extract inv_quad solves from all the solves
            if self.inv_quad:
                inv_quad_solves = solves.narrow(-1, num_random_probes, num_inv_quad_solves)
                inv_quad_term = (inv_quad_solves * inv_quad_rhs).sum(-1).sum(-1, keepdim=(self.batch_size is None))

            self.matmul_closure = matmul_closure
            self.num_random_probes = num_random_probes
            self.num_inv_quad_solves = num_inv_quad_solves
            self.solves = solves
            self.save_for_backward(*closure_args)

            return inv_quad_term, log_det_term

        def backward(self, inv_quad_grad_output, log_det_grad_output):
            if derivative_quadratic_form_factory is None:
                raise NotImplementedError
            closure_arg_grads = None
            inv_quad_rhs_grad = None

            # Which backward passes should we compute?
            compute_inv_quad_grad = inv_quad_grad_output.sum() and self.inv_quad
            compute_log_det_grad = log_det_grad_output.sum() and self.log_det

            # Get input arguments, and get gradients in the proper form
            closure_args = None
            closure_args = self.saved_tensors
            if self.inv_quad:
                inv_quad_grad_output = inv_quad_grad_output.unsqueeze(-1)
                if self.batch_size is not None:
                    inv_quad_grad_output.unsqueeze_(-1)
            if compute_log_det_grad:
                log_det_grad_output = log_det_grad_output.unsqueeze(-1)
                if self.batch_size is not None:
                    log_det_grad_output.unsqueeze_(-1)

            # Divide up the solves
            probe_vector_solves = None
            probe_vectors = None
            inv_quad_solves = None
            neg_inv_quad_solves_times_grad_out = None
            if compute_log_det_grad:
                coef = 1. / self.probe_vectors.size(-1)
                probe_vector_solves = self.solves.narrow(-1, 0, self.num_random_probes).mul(coef)
                probe_vector_solves.mul_(self.probe_vector_norms).mul_(log_det_grad_output)
                probe_vectors = self.probe_vectors
            if self.inv_quad:
                inv_quad_solves = self.solves.narrow(-1, self.num_random_probes, self.num_inv_quad_solves)
                neg_inv_quad_solves_times_grad_out = inv_quad_solves.mul(inv_quad_grad_output).mul_(-1)

            # input_1 gradient
            if any(self.needs_input_grad[:len(closure_args)]):
                # Collect terms for arg grads
                left_factors_list = []
                right_factors_list = []

                if compute_log_det_grad:
                    left_factors_list.append(probe_vector_solves)
                    right_factors_list.append(probe_vectors)

                if compute_inv_quad_grad:
                    left_factors_list.append(neg_inv_quad_solves_times_grad_out)
                    right_factors_list.append(inv_quad_solves)

                left_factors = torch.cat(left_factors_list, -1).transpose(-1, -2)
                right_factors = torch.cat(right_factors_list, -1).transpose(-1, -2)
                closure_arg_grads = list(derivative_quadratic_form_factory(*closure_args)(left_factors,
                                                                                          right_factors))

            # input_2 gradient
            if compute_inv_quad_grad and self.needs_input_grad[-1]:
                inv_quad_rhs_grad = neg_inv_quad_solves_times_grad_out.mul_(-2)
            elif self.inv_quad:
                inv_quad_rhs_grad = inv_quad_solves.new(*inv_quad_solves.size()).zero_()
            if self.is_vector:
                inv_quad_rhs_grad.squeeze_(-1)

            res = tuple(closure_arg_grads + [inv_quad_rhs_grad])
            return res

    return InvQuadLogDet


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
            def is_empty(tensor):
                return tensor.numel() == 0 or (tensor.numel() == 1 and tensor[0] == 0)

            # Taken from http://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf
            if any(self.needs_input_grad):
                args = self.saved_tensors
                if is_empty(root_grad_output):
                    root_grad_output = None
                if is_empty(inverse_grad_output):
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
