import torch


class StochasticLQ(object):
    """
    Implements an approximate log determinant calculation for symmetric positive definite matrices
    using stochastic Lanczos quadrature. For efficient calculation of derivatives, We additionally
    compute the trace of the inverse using the same probe vector the log determinant was computed
    with. For more details, see Dong et al. 2017 (in submission).
    """
    def __init__(self, cls=None, max_iter=15, num_random_probes=10):
        """
        The nature of stochastic Lanczos quadrature is that the calculation of tr(f(A)) is both inaccurate and
        stochastic. An instance of StochasticLQ has two parameters that control these tradeoffs. Increasing either
        parameter increases the running time of the algorithm.
        Args:
            cls - Tensor constructor - to ensure correct type (default - default tensor)
            max_iter (scalar) - The number of Lanczos iterations to perform. Increasing this makes the estimate of
                     tr(f(A)) more accurate in expectation -- that is, the average value returned has lower error.
            num_random_probes (scalar) - The number of random probes to use in the stochastic trace estimation.
                              Increasing this makes the estimate of tr(f(A)) lower variance -- that is, the value
                              returned is more consistent.
        """
        self.cls = cls or torch.Tensor
        self.max_iter = max_iter
        self.num_random_probes = num_random_probes

    def lanczos_batch(self, matmul_closure, rhs_vectors):
        is_batch = rhs_vectors.ndimension() == 3
        if is_batch:
            batch, dim, num_vectors = rhs_vectors.size()
            num_iters = min(self.max_iter, dim)

            Q = self.cls(batch, num_vectors, dim, num_iters).zero_()
            alpha = self.cls(batch, num_vectors, num_iters).zero_()
            beta = self.cls(batch, num_vectors, num_iters).zero_()

            rhs_vectors = rhs_vectors / torch.norm(rhs_vectors, 2, dim=1).unsqueeze(1)
            Q[:, :, :, 0] = rhs_vectors.transpose(1, 2)
            U = rhs_vectors

            R = matmul_closure(U)
            a = U.mul(R).sum(1).unsqueeze(1)

            rhs_vectors = (R - a * U) + 1e-10

            beta[:, :, 0] = torch.norm(rhs_vectors, 2, dim=1)
            alpha[:, :, 0] = a

            for k in range(1, num_iters):
                U, rhs_vectors, alpha_k, beta_k = self._lanczos_step_batch(U, rhs_vectors,
                                                                           matmul_closure, Q[:, :, :, :k])

                alpha[:, :, k] = alpha_k
                beta[:, :, k] = beta_k
                Q[:, :, :, k] = U.transpose(1, 2)

                if all(torch.abs(beta[:, :, k]).view(-1) < 1e-4) or all(torch.abs(alpha[:, :, k]).view(-1) < 1e-4):
                    break

            if k == 1:
                Ts = alpha[:, :, :k].unsqueeze(1)
                Qs = Q[:, :, :, :k]
            else:
                alpha = alpha[:, :, :k]
                beta = beta[:, :, 1:k]

                Qs = Q[:, :, :, :k]

                Ts = self.cls(batch, num_vectors, k, k)
                for i in range(num_vectors):
                    for j in xrange(batch):
                        Ts[j, i, :, :] = torch.diag(alpha[j, i, :]) + torch.diag(beta[j, i, :], 1) + \
                            torch.diag(beta[j, i, :], -1)

            return Qs, Ts

        dim, num_vectors = rhs_vectors.size()
        num_iters = min(self.max_iter, dim)

        Q = self.cls(num_vectors, dim, num_iters).zero_()
        alpha = self.cls(num_vectors, num_iters).zero_()
        beta = self.cls(num_vectors, num_iters).zero_()

        rhs_vectors = rhs_vectors / torch.norm(rhs_vectors, 2, dim=0)
        Q[:, :, 0] = rhs_vectors.t()
        U = rhs_vectors

        R = matmul_closure(U)
        a = U.mul(R).sum(0)

        rhs_vectors = (R - a * U) + 1e-10

        beta[:, 0] = torch.norm(rhs_vectors, 2, dim=0)
        alpha[:, 0] = a

        if num_iters == 1:
            Ts = alpha[:, :].unsqueeze(1)
            Qs = Q[:, :, :]

        else:
            for k in range(1, num_iters):
                U, rhs_vectors, alpha_k, beta_k = self._lanczos_step_batch(U, rhs_vectors, matmul_closure, Q[:, :, :k])

                alpha[:, k] = alpha_k
                beta[:, k] = beta_k
                Q[:, :, k] = U.t()

                if not torch.sum(torch.abs(beta[:, k]) > 1e-4) or not torch.sum(torch.abs(alpha[:, k]) > 1e-4):
                    break

            alpha = alpha[:, :k + 1]
            beta = beta[:, 1:k + 1]

            Qs = Q[:, :, :k + 1]

            Ts = self.cls(num_vectors, k + 1, k + 1)
            for i in range(num_vectors):
                Ts[i, :, :] = torch.diag(alpha[i, :]) + torch.diag(beta[i, :], 1) + torch.diag(beta[i, :], -1)

        return Qs, Ts

    def _lanczos_step_batch(self, U, rhs_vectors, matmul_closure, Q):
        if Q.ndimension() == 4:
            batch_size, num_vectors, dim, num_iters = Q.size()
            norm_vs = torch.norm(rhs_vectors, 2, dim=1).unsqueeze(1)
            orig_U = U

            U = rhs_vectors / norm_vs

            U = U - self._batch_mv(Q, self._batch_mv(Q.transpose(2, 3), U.transpose(1, 2))).transpose(1, 2)

            U = U / torch.norm(U, 2, dim=1).unsqueeze(1)

            R = matmul_closure(U) - norm_vs * orig_U

            a = U.mul(R).sum(1).unsqueeze(1)

            rhs_vectors = (R - a * U)

            # Numerical Problems
            rhs_vectors += 1e-10
            a = torch.max(a, a.new(*a.size()).fill_(1) * 1e-20)
            norm_vs = torch.max(norm_vs, norm_vs.new(*norm_vs.size()).fill_(1) * 1e-20)
            U += 1e-10

            return U, rhs_vectors, a, norm_vs

        num_vectors, dim, num_iters = Q.size()
        norm_vs = torch.norm(rhs_vectors, 2, dim=0)
        orig_U = U

        U = rhs_vectors / norm_vs

        U = U - self._batch_mv(Q, self._batch_mv(Q.transpose(1, 2), U.t())).t()

        U = U / torch.norm(U, 2, dim=0)

        R = matmul_closure(U) - norm_vs * orig_U

        a = U.mul(R).sum(0)

        rhs_vectors = (R - a * U)

        # Numerical Problems
        rhs_vectors += 1e-10
        a = torch.max(a, a.new(*a.size()).fill_(1) * 1e-20)
        norm_vs = torch.max(norm_vs, norm_vs.new(*norm_vs.size()).fill_(1) * 1e-20)
        U += 1e-10

        return U, rhs_vectors, a, norm_vs

    def _batch_mv(self, M, V):
        if M.ndimension() == 4:
            batch_size, num, n, m = M.size()
            V_expand = V.unsqueeze(1).expand(batch_size, n, num, m).transpose(1, 2)
            return (M * V_expand).sum(3)

        num, n, m = M.size()
        V_expand = V.expand(n, num, m).transpose(0, 1)
        return (M * V_expand).sum(2)

    def binary_search_symeig(self, T):
        left = 0
        right = len(T)
        while right - left > 1:
            mid = (left + right) // 2
            eigs = T[:mid, :mid].symeig()[0]
            if torch.min(eigs) < -1e-4:
                right = mid - 1
            else:
                left = mid

        return left

    def evaluate(self, matmul_closure, n, funcs, batch_size=-1):
        """
        Computes tr(f(A)) for an arbitrary list of functions, where f(A) is equivalent to applying the function
        elementwise to the eigenvalues of A, i.e., if A = V\LambdaV^{T}, then f(A) = Vf(\Lambda)V^{T}, where
        f(\Lambda) is applied elementwise.
        Note that calling this function with a list of functions to apply is significantly more efficient than
        calling it multiple times with one function -- each additional function after the first requires negligible
        additional computation.
        Args:
            - A (matrix n x n or closure) - Either the input matrix A or a closure that takes an n dimensional vector v
                and returns Av.
            - n (scalar) - dimension of the matrix A. We require this because, if A is passed in as a closure, we
                have no good way of determining the size of A.
            - funcs (list of closures) - A list of functions [f_1,...,f_k]. tr(f_i(A)) is computed for each function.
                    Each function in the closure should expect to take a torch vector of eigenvalues as input and apply
                    the function elementwise. For example, to compute logdet(A) = tr(log(A)), [lambda x: x.log()] would
                    be a reasonable value of funcs.
        Returns:
            - results (list of scalars) - The trace of each supplied function applied to the matrix, e.g.,
                      [tr(f_1(A)),tr(f_2(A)),...,tr(f_k(A))].
        """
        if torch.is_tensor(matmul_closure):
            lhs = matmul_closure
            if lhs.numel() == 1:
                return [func(lhs).squeeze()[0] for func in funcs]

            def default_matmul_closure(tensor):
                return lhs.matmul(tensor)
            matmul_closure = default_matmul_closure

        if batch_size != -1:
            V = self.cls(n, self.num_random_probes).bernoulli_().mul_(2).add_(-1)
            V.div_(torch.norm(V, 2, dim=0).expand_as(V))
            V = V.unsqueeze(0).expand(batch_size, n, self.num_random_probes)
        else:
            V = self.cls(n, self.num_random_probes).bernoulli_().mul_(2).add_(-1)
            V.div_(torch.norm(V, 2, dim=0).expand_as(V))

        if V.ndimension() == 3:
            results = [V.new(V.size(0)).zero_()] * len(funcs)
        else:
            results = [0] * len(funcs)

        _, Ts = self.lanczos_batch(matmul_closure, V)
        Ts = Ts.cpu()

        for j in range(self.num_random_probes):
            if Ts.ndimension() == 4:
                for k in xrange(Ts.size(0)):
                    T = Ts[k, j, :, :]

                    [f, Y] = T.symeig(eigenvectors=True)
                    if min(f) < -1e-4:
                        last_proper = max(self.binary_search_symeig(T), 1)
                        [f, Y] = T[:last_proper, :last_proper].symeig(eigenvectors=True)

                    for i, func in enumerate(funcs):
                        results[i][k] = results[i][k] + n / float(self.num_random_probes) * \
                            (Y[0, :].pow(2).dot(func(f + 1.1e-4)))

            else:
                T = Ts[j, :, :]

                [f, Y] = T.symeig(eigenvectors=True)
                if min(f) < -1e-4:
                    last_proper = max(self.binary_search_symeig(T), 1)
                    [f, Y] = T[:last_proper, :last_proper].symeig(eigenvectors=True)

                for i, func in enumerate(funcs):
                    results[i] = results[i] + n / float(self.num_random_probes) * (Y[0, :].pow(2).dot(func(f + 1.1e-4)))

        return results
