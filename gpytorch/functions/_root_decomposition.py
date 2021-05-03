#!/usr/bin/env python3

import torch
from torch.autograd import Function

from .. import settings
from ..utils import lanczos


class RootDecomposition(Function):
    @staticmethod
    def forward(
        ctx,
        representation_tree,
        max_iter,
        dtype,
        device,
        batch_shape,
        matrix_shape,
        root,
        inverse,
        initial_vectors,
        *matrix_args,
    ):
        r"""
        :param list matrix_args: The arguments representing the symmetric matrix A (or batch of PSD matrices A)

        :rtype: (torch.Tensor, torch.Tensor)
        :return: :attr:`R`, such that :math:`R R^T \approx A`, and :attr:`R_inv`, such that
            :math:`R_{inv} R_{inv}^T \approx A^{-1}` (will only be populated if self.inverse = True)
        """
        from ..lazy import lazify

        ctx.representation_tree = representation_tree
        ctx.device = device
        ctx.dtype = dtype
        ctx.matrix_shape = matrix_shape
        ctx.max_iter = max_iter
        ctx.batch_shape = batch_shape
        ctx.root = root
        ctx.inverse = inverse
        ctx.initial_vectors = initial_vectors

        # Get closure for matmul
        lazy_tsr = ctx.representation_tree(*matrix_args)
        matmul_closure = lazy_tsr._matmul
        # Do lanczos
        q_mat, t_mat = lanczos.lanczos_tridiag(
            matmul_closure,
            ctx.max_iter,
            dtype=ctx.dtype,
            device=ctx.device,
            matrix_shape=ctx.matrix_shape,
            batch_shape=ctx.batch_shape,
            init_vecs=ctx.initial_vectors,
        )

        if ctx.batch_shape is None:
            q_mat = q_mat.unsqueeze(-3)
            t_mat = t_mat.unsqueeze(-3)
        if t_mat.ndimension() == 3:  # If we only used one probe vector
            q_mat = q_mat.unsqueeze(0)
            t_mat = t_mat.unsqueeze(0)
        n_probes = t_mat.size(0)

        mins = lazify(t_mat).diag().min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        jitter_mat = (settings.tridiagonal_jitter.value() * mins) * torch.eye(
            t_mat.size(-1), device=t_mat.device, dtype=t_mat.dtype
        ).expand_as(t_mat)
        eigenvalues, eigenvectors = lanczos.lanczos_tridiag_to_diag(t_mat + jitter_mat)

        # Get orthogonal matrix and eigenvalue roots
        q_mat = q_mat.matmul(eigenvectors)
        root_evals = eigenvalues.sqrt()

        # Store q_mat * t_mat_chol
        # Decide if we're computing the inverse, or the regular root
        root = torch.empty(0, dtype=q_mat.dtype, device=q_mat.device)
        inverse = torch.empty(0, dtype=q_mat.dtype, device=q_mat.device)
        if ctx.inverse:
            inverse = q_mat / root_evals.unsqueeze(-2)
        if ctx.root:
            root = q_mat * root_evals.unsqueeze(-2)

        if settings.memory_efficient.off():
            ctx._lazy_tsr = lazy_tsr

        if ctx.batch_shape is None:
            root = root.squeeze(1) if root.numel() else root
            q_mat = q_mat.squeeze(1)
            root_evals = root_evals.squeeze(1)
            inverse = inverse.squeeze(1) if inverse.numel() else inverse
        if n_probes == 1:
            root = root.squeeze(0) if root.numel() else root
            q_mat = q_mat.squeeze(0)
            root_evals = root_evals.squeeze(0)
            inverse = inverse.squeeze(0) if inverse.numel() else inverse

        to_save = list(matrix_args) + [q_mat, root_evals, inverse]
        ctx.save_for_backward(*to_save)
        return root, inverse

    @staticmethod
    def backward(ctx, root_grad_output, inverse_grad_output):
        # Taken from http://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf
        if any(ctx.needs_input_grad):

            def is_empty(tensor):
                return tensor.numel() == 0 or (tensor.numel() == 1 and tensor[0] == 0)

            # Fix outputs and gradients
            if is_empty(root_grad_output):
                root_grad_output = None
            if is_empty(inverse_grad_output):
                inverse_grad_output = None

            # Get saved tensors
            matrix_args = ctx.saved_tensors[:-3]
            q_mat = ctx.saved_tensors[-3]
            root_evals = ctx.saved_tensors[-2]
            inverse = ctx.saved_tensors[-1]
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
            if hasattr(ctx, "_lazy_tsr"):
                lazy_tsr = ctx._lazy_tsr
            else:
                lazy_tsr = ctx.representation_tree(*matrix_args)

            # Get root inverse
            if not ctx.inverse:
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

            return tuple([None] * 9 + list(res))
        else:
            pass
