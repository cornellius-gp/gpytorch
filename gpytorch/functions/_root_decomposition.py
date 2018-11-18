#!/usr/bin/env python3

import torch
from torch.autograd import Function
from ..utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from .. import settings


class RootDecomposition(Function):
    def __init__(
        self,
        representation_tree,
        max_iter,
        dtype,
        device,
        matrix_shape,
        batch_shape,
        root=True,
        inverse=False,
        initial_vectors=None,
    ):
        self.representation_tree = representation_tree
        self.device = device
        self.dtype = dtype
        self.matrix_shape = matrix_shape
        self.max_iter = max_iter
        self.batch_shape = batch_shape
        self.root = root
        self.inverse = inverse
        self.initial_vectors = initial_vectors

    def forward(self, *matrix_args):
        """
        *matrix_args - The arguments representing the symmetric matrix A (or batch of PSD matrices A)

        Returns:
        - (Tensor) R, such that R R^T \approx A
        - (Tensor) R_inv, such that R_inv R_inv^T \approx A^{-1} (will only be populated if self.inverse = True)
        """
        # Get closure for matmul
        lazy_tsr = self.representation_tree(*matrix_args)
        matmul_closure = lazy_tsr._matmul
        # Do lanczos
        q_mat, t_mat = lanczos_tridiag(
            matmul_closure,
            self.max_iter,
            dtype=self.dtype,
            device=self.device,
            matrix_shape=self.matrix_shape,
            batch_shape=self.batch_shape,
            init_vecs=self.initial_vectors,
        )
        if self.batch_shape is None:
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
        root = torch.empty(0, dtype=q_mat.dtype, device=q_mat.device)
        inverse = torch.empty(0, dtype=q_mat.dtype, device=q_mat.device)
        if self.inverse:
            inverse = q_mat / root_evals.unsqueeze(-2)
        if self.root:
            root = q_mat * root_evals.unsqueeze(-2)

        if not settings.memory_efficient.on():
            self._lazy_tsr = lazy_tsr

        if self.batch_shape is None:
            root = root.squeeze(1) if root.numel() else root
            q_mat = q_mat.squeeze(1)
            t_mat = t_mat.squeeze(1)
            root_evals = root_evals.squeeze(1)
            inverse = inverse.squeeze(1) if inverse.numel() else inverse
        if n_probes == 1:
            root = root.squeeze(0) if root.numel() else root
            q_mat = q_mat.squeeze(0)
            t_mat = t_mat.squeeze(0)
            root_evals = root_evals.squeeze(0)
            inverse = inverse.squeeze(0) if inverse.numel() else inverse

        to_save = list(matrix_args) + [q_mat, root_evals, inverse]
        self.save_for_backward(*to_save)
        return root, inverse

    def backward(self, root_grad_output, inverse_grad_output):
        # Taken from http://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf
        if any(self.needs_input_grad):

            def is_empty(tensor):
                return tensor.numel() == 0 or (tensor.numel() == 1 and tensor[0] == 0)

            # Fix outputs and gradients
            if is_empty(root_grad_output):
                root_grad_output = None
            if is_empty(inverse_grad_output):
                inverse_grad_output = None

            # Get saved tensors
            matrix_args = self.saved_tensors[:-3]
            q_mat = self.saved_tensors[-3]
            root_evals = self.saved_tensors[-2]
            inverse = self.saved_tensors[-1]
            is_batch = False

            if root_grad_output is not None:
                if root_grad_output.ndimension() == 2 and q_mat.ndimension() > 2:
                    root_grad_output = root_grad_output.unsqueeze(0)
                    is_batch = True
                if root_grad_output.ndimension() == 3 and q_mat.ndimension() > 3:
                    root_grad_output = root_grad_output.unsqueeze(0)
                    is_batch = True
            if inverse_grad_output is not None:
                if inverse_grad_output.ndimension() == 2 and q_mat.ndimension() > 2:
                    inverse_grad_output = inverse_grad_output.unsqueeze(0)
                    is_batch = True
                if inverse_grad_output.ndimension() == 3 and q_mat.ndimension() > 3:
                    inverse_grad_output = inverse_grad_output.unsqueeze(0)
                    is_batch = True

            # Get closure for matmul
            if hasattr(self, "_lazy_tsr"):
                lazy_tsr = self._lazy_tsr
            else:
                lazy_tsr = self.representation_tree(*matrix_args)

            # Get root inverse
            if not self.inverse:
                inverse = q_mat / root_evals.unsqueeze(-2)
            # Left factor:
            left_factor = torch.zeros_like(inverse)
            if root_grad_output is not None:
                left_factor.add_(root_grad_output)
            if inverse_grad_output is not None:
                # -root^-T grad_output.T root^-T
                left_factor.sub_(torch.matmul(inverse, inverse_grad_output.transpose(-1, -2)).matmul(inverse))

            # Right factor
            right_factor = inverse.div(2.0)

            # Fix batches
            if is_batch:
                left_factor = left_factor.permute(1, 0, 2, 3).contiguous()
                left_factor = left_factor.view(inverse.size(1), -1, left_factor.size(-1))
                right_factor = right_factor.permute(1, 0, 2, 3).contiguous()
                right_factor = right_factor.view(inverse.size(1), -1, right_factor.size(-1))
            else:
                left_factor = left_factor.contiguous()
                right_factor = right_factor.contiguous()
            res = lazy_tsr._quad_form_derivative(left_factor, right_factor)

            return tuple(res)
        else:
            pass
