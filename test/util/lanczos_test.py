import torch
from gpytorch.utils import approx_equal


def lanczos(matmul_closure, max_iter, eps=1e-10, init_vecs=None,
            tensor_cls=None, batch_size=None, n_dims=None, n_init_vecs=1):
    # Determine batch mode
    is_batch = False

    # Get initial probe ectors - and define if not available
    if init_vecs is None:
        if tensor_cls is None:
            raise RuntimeError('You must supply tensor_cls if you don\'t supply init_vecs')
        if n_dims is None:
            raise RuntimeError('You must supply n_dims if you don\'t supply init_vecs')

        init_vecs = tensor_cls(n_dims, n_init_vecs).normal_()
        init_vecs, _ = torch.qr(init_vecs)
        init_vecs.unsqueeze_(0)
        if batch_size is not None:
            is_batch = True
            init_vecs = init_vecs.expand(batch_size, n_dims, n_init_vecs)
        else:
            batch_size = 1
    else:
        tensor_cls = init_vecs.new
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
    alpha = tensor_cls(n_iter, batch_size, n_init_vecs).zero_()
    beta = tensor_cls(n_iter, batch_size, n_init_vecs).zero_()

    # Begin algorithm
    # Initial Q vector: q_0_vec
    q_0_vec = init_vecs / torch.norm(init_vecs, 2, dim=dim_dimension).unsqueeze(dim_dimension)

    # Initial alpha value: alpha_0
    r_vec = batch_aware_matmul_closure(q_0_vec)
    alpha_0 = q_0_vec.mul(r_vec).sum(dim_dimension)

    # Initial beta value: beta_0
    r_vec.sub_(alpha_0.unsqueeze(dim_dimension).mul(q_0_vec)).add_(eps)
    beta_0 = torch.norm(r_vec, 2, dim=dim_dimension)

    # Copy over q_0_vec, alpha_0, beta_0
    q_mat[0].copy_(q_0_vec)
    alpha[0].copy_(alpha_0)
    beta[0].copy_(beta_0)

    # Now we start the iteration
    for k in range(1, n_iter):
        # Get previous values
        q_prev_vec = q_mat[k - 1]
        beta_prev = beta[k - 1].unsqueeze(dim_dimension)

        # Compute next vector of Q
        q_curr_vec = r_vec.div_(beta_prev)

        # Compute next alpha value
        r_vec = batch_aware_matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)
        alpha_curr = q_curr_vec.mul(r_vec).sum(dim_dimension)

        # Compute next beta value
        r_vec.sub_(alpha_curr.unsqueeze(dim_dimension).mul(q_curr_vec))
        beta_curr = torch.norm(r_vec, 2, dim=dim_dimension)

        # Numerical Problems
        r_vec.add_(eps)
        q_curr_vec.add_(eps)
        alpha_curr.clamp_(eps, 1e100)
        beta_curr.clamp_(eps, 1e100)

        # Copy over q_curr_vec, alpha_curr, beta_curr
        q_mat[k].copy_(q_curr_vec.clone())
        beta[k].copy_(beta_curr.clone())
        alpha[k].copy_(alpha_curr.clone())

        if torch.sum(beta_curr.abs() > 1e-4) == 0:
            break

    # Now let's transpose q_mat, alpha, and beta into the correct shape
    n_iter = k + 1
    q_mat = q_mat[:n_iter].permute(3, 1, 2, 0).contiguous()  # n_init_vecs x batch_size x n_dims x n_iter
    alpha = alpha[:n_iter].permute(2, 1, 0)  # n_init_vecs x batch_size x n_iter
    if k > 1:
        beta = beta[:n_iter - 1].permute(2, 1, 0)  # n_init_vecs x batch_size x n_iter
    else:
        beta = None

    # Make alpha, beta into a tridiagonal matrix
    batch_index = tensor_cls(n_init_vecs * batch_size).long()
    torch.arange(0, n_init_vecs * batch_size, out=batch_index)
    off_batch_index = batch_index.unsqueeze(1).repeat(n_iter - 1, 1).view(-1)
    batch_index = batch_index.unsqueeze(1).repeat(n_iter, 1).view(-1)

    diag_index = tensor_cls(n_iter).long()
    torch.arange(0, n_iter, out=diag_index)
    diag_index = diag_index.unsqueeze(1).repeat(1, n_init_vecs * batch_size).view(-1)
    off_diag_index = tensor_cls(n_iter - 1).long()
    torch.arange(0, n_iter - 1, out=off_diag_index)
    off_diag_index = off_diag_index.unsqueeze(1).repeat(1, n_init_vecs * batch_size).view(-1)

    main_flattened_indices = sum([
        batch_index * (n_init_vecs * batch_size * n_iter),
        diag_index * (n_iter + 1),
    ])
    lower_off_flattened_indices = sum([
        off_batch_index * (n_init_vecs * batch_size * (n_iter - 1)),
        (off_diag_index + 1) * n_iter,
        off_diag_index,
    ])
    upper_off_flattened_indices = sum([
        off_batch_index * (n_init_vecs * batch_size * (n_iter - 1)),
        off_diag_index * n_iter,
        off_diag_index + 1,
    ])

    # Now let's copy over alpha, beta values!
    t_mat = tensor_cls(n_init_vecs, batch_size, n_iter, n_iter).zero_()
    t_mat.view(-1).index_copy_(0, main_flattened_indices, alpha.view(-1))
    if beta is not None:
        t_mat.view(-1).index_copy_(0, lower_off_flattened_indices, beta.view(-1))
        t_mat.view(-1).index_copy_(0, upper_off_flattened_indices, beta.view(-1))

    # If we weren't in batch mode, remove batch dimension
    if not is_batch:
        q_mat.squeeze_(batch_dimension)
        t_mat.squeeze_(batch_dimension)

    # We're done!
    return q_mat, t_mat


def test_lanczos():
    matrix = torch.randn(5, 5)
    matrix = matrix.matmul(matrix.transpose(-1, -2))
    matrix.div_(matrix.norm())
    matrix.add_(torch.ones(matrix.size(-1)).mul(1e-6).diag())
    q_mat, t_mat = lanczos(matrix.matmul, max_iter=32, tensor_cls=matrix.new, n_dims=matrix.size(-1))

    q_mat.squeeze_(0)
    t_mat.squeeze_(0)

    approx = q_mat.matmul(t_mat).matmul(q_mat.transpose(-1, -2))
    assert approx_equal(approx, matrix)
