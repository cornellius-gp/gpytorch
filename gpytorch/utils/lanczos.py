from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


def lanczos_tridiag(matmul_closure, max_iter, tol=1e-5, init_vecs=None,
                    tensor_cls=None, batch_size=None, n_dims=None, n_init_vecs=None):
    # Determine batch mode
    is_batch = False
    multiple_init_vecs = False
    if torch.is_tensor(matmul_closure):
        lhs = matmul_closure

        def default_matmul_closure(tensor):
            return lhs.matmul(tensor)
        matmul_closure = default_matmul_closure
    # Get initial probe ectors - and define if not available
    if init_vecs is None:
        if tensor_cls is None:
            raise RuntimeError('You must supply tensor_cls if you don\'t supply init_vecs')
        if n_dims is None:
            raise RuntimeError('You must supply n_dims if you don\'t supply init_vecs')

        if batch_size is not None:
            is_batch = True
        else:
            batch_size = 1
        if n_init_vecs is not None:
            multiple_init_vecs = True
        else:
            n_init_vecs = 1

        init_vecs = tensor_cls(1, n_dims, n_init_vecs).normal_()
        init_vecs = init_vecs.expand(batch_size, n_dims, n_init_vecs)
    else:
        tensor_cls = init_vecs.new
        if n_init_vecs is not None or init_vecs.size(-1) > 1:
            multiple_init_vecs = True
        if init_vecs.ndimension() == 3:
            batch_size, n_dims, n_init_vecs = init_vecs.size()
            is_batch = True
        else:
            init_vecs = init_vecs.unsqueeze(0)
            batch_size, n_dims, n_init_vecs = init_vecs.size()

    # Modify matmul closure so that it handles batch vs non-batch
    def batch_aware_matmul_closure(tensor):
        if is_batch:
            return matmul_closure(tensor)
        else:
            return matmul_closure(tensor.squeeze(0)).unsqueeze_(0)

    # Define some constants
    n_iter = min(max_iter, n_dims)
    batch_dimension = 1
    dim_dimension = -2

    # Create storage for q_mat, alpha,and beta
    # q_mat - batch version of Q - orthogonal matrix of decomp
    # alpha - batch version main diagonal of T
    # beta - batch version of off diagonal of T
    q_mat = tensor_cls(n_iter, batch_size, n_dims, n_init_vecs).zero_()
    t_mat = tensor_cls(n_iter, n_iter, batch_size, n_init_vecs).zero_()

    # Begin algorithm
    # Initial Q vector: q_0_vec
    q_0_vec = init_vecs / torch.norm(init_vecs, 2, dim=dim_dimension).unsqueeze(dim_dimension)
    q_mat[0].copy_(q_0_vec)

    # Initial alpha value: alpha_0
    r_vec = batch_aware_matmul_closure(q_0_vec)
    alpha_0 = q_0_vec.mul(r_vec).sum(dim_dimension)

    # Initial beta value: beta_0
    r_vec.sub_(alpha_0.unsqueeze(dim_dimension).mul(q_0_vec))
    beta_0 = torch.norm(r_vec, 2, dim=dim_dimension)

    # Copy over alpha_0 and beta_0 to t_mat
    t_mat[0, 0].copy_(alpha_0)
    t_mat[0, 1].copy_(beta_0)
    t_mat[1, 0].copy_(beta_0)

    # Compute teh first new vector
    q_mat[1].copy_(r_vec.div_(beta_0.unsqueeze(dim_dimension)))

    # Now we start the iteration
    for k in range(1, n_iter):
        # Get previous values
        q_prev_vec = q_mat[k - 1]
        q_curr_vec = q_mat[k]
        beta_prev = t_mat[k, k - 1].unsqueeze(dim_dimension)

        # Compute next alpha value
        r_vec = batch_aware_matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)
        alpha_curr = q_curr_vec.mul(r_vec).sum(dim_dimension, keepdim=True)
        # Copy over to t_mat
        t_mat[k, k].copy_(alpha_curr.squeeze(dim_dimension))

        # Copy over alpha_curr, beta_curr to t_mat
        if (k + 1) < n_iter:
            # Compute next residual value
            r_vec.sub_(alpha_curr.mul(q_curr_vec))
            # Full reorthogonalization: r <- r - Q (Q^T r)
            correction = r_vec.unsqueeze(0).mul(q_mat[:k + 1]).sum(dim_dimension, keepdim=True)
            correction = q_mat[:k + 1].mul(correction).sum(0)
            r_vec.sub_(correction)
            r_vec_norm = torch.norm(r_vec, 2, dim=dim_dimension, keepdim=True)
            r_vec.div_(r_vec_norm)

            # Get next beta value
            beta_curr = r_vec_norm.squeeze_(dim_dimension)
            # Update t_mat with new beta value
            t_mat[k, k + 1].copy_(beta_curr)
            t_mat[k + 1, k].copy_(beta_curr)

            # Run more reorthoganilzation if necessary
            inner_products = q_mat[:k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)
            could_reorthogonalize = False
            for i in range(10):
                if not torch.sum(inner_products > tol):
                    could_reorthogonalize = True
                    break
                correction = r_vec.unsqueeze(0).mul(q_mat[:k + 1]).sum(dim_dimension, keepdim=True)
                correction = q_mat[:k + 1].mul(correction).sum(0)
                r_vec.sub_(correction)
                r_vec_norm = torch.norm(r_vec, 2, dim=dim_dimension, keepdim=True)
                r_vec.div_(r_vec_norm)
                inner_products = q_mat[:k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)

            # Update q_mat with new q value
            q_mat[k + 1].copy_(r_vec)

            if torch.sum(beta_curr.abs() > 1e-6) == 0 or not could_reorthogonalize:
                break

    # Now let's transpose q_mat, t_mat intot the correct shape
    n_iter = k + 1

    # n_init_vecs x batch_size x n_dims x n_iter
    q_mat = q_mat[:n_iter + 1].permute(3, 1, 2, 0).contiguous()
    # n_init_vecs x batch_size x n_iter x n_iter
    t_mat = t_mat[:n_iter + 1, :n_iter + 1].permute(3, 2, 0, 1).contiguous()

    # If we weren't in batch mode, remove batch dimension
    if not is_batch:
        q_mat.squeeze_(batch_dimension)
        t_mat.squeeze_(batch_dimension)
    if not multiple_init_vecs:
        q_mat.squeeze_(0)
        t_mat.squeeze_(0)

    # We're done!
    return q_mat, t_mat


def lanczos_tridiag_to_diag(t_mat):
    """
    Given a num_init_vecs x num_batch x k x k tridiagonal matrix t_mat,
    returns a num_init_vecs x num_batch x k set of eigenvalues
    and a num_init_vecs x num_batch x k x k set of eigenvectors.

    TODO: make the eigenvalue computations done in batch mode.
    """
    t_mat_orig = t_mat
    t_mat = t_mat.cpu()

    if t_mat.dim() == 3:
        t_mat = t_mat.unsqueeze(0)
    batch_dim1 = t_mat.size(0)
    batch_dim2 = t_mat.size(1)
    n = t_mat.size(2)

    eigenvectors = t_mat.new(*t_mat.shape)
    eigenvalues = t_mat.new(batch_dim1, batch_dim2, n)

    for i in range(batch_dim1):
        for j in range(batch_dim2):
            evals, evecs = t_mat[i, j].symeig(eigenvectors=True)
            mask = evals.ge(0)
            eigenvectors[i, j] = evecs * mask.type_as(evecs).unsqueeze(0)
            eigenvalues[i, j] = evals.masked_fill_(1 - mask, 1)

    return eigenvalues.type_as(t_mat_orig), eigenvectors.type_as(t_mat_orig)
