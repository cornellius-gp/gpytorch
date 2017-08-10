import torch
import math


class StochasticLQ(object):
    """
    Implements an approximate log determinant calculation for symmetric positive definite matrices
    using stochastic Lanczos quadrature. For efficient calculation of derivatives, We additionally
    compute the trace of the inverse using the same probe vector the log determinant was computed
    with. For more details, see Dong et al. 2017 (in submission).
    """
    def __init__(self, max_iter=15, num_random_probes=10):
        """
        The nature of stochastic Lanczos quadrature is that the calculation of tr(f(A)) is both inaccurate and
        stochastic. An instance of StochasticLQ has two parameters that control these tradeoffs. Increasing either
        parameter increases the running time of the algorithm.
        Args:
            max_iter (scalar) - The number of Lanczos iterations to perform. Increasing this makes the estimate of
                     tr(f(A)) more accurate in expectation -- that is, the average value returned has lower error.
            num_random_probes (scalar) - The number of random probes to use in the stochastic trace estimation.
                              Increasing this makes the estimate of tr(f(A)) lower variance -- that is, the value
                              returned is more consistent.
        """
        self.max_iter = max_iter
        self.num_random_probes = num_random_probes

    def lanczos(self, mv_closure, b):
        """
        Performs self.max_iter (at most n) iterations of the Lanczos iteration to decompose A as AQ = QT
        with Q an orthogonal basis for the Krylov subspace [Ab,A^{2}b,...,A^{max_iter}b], and T tridiagonal.
        """
        n = len(b)
        num_iters = min(self.max_iter, n)

        Q = torch.zeros(n, num_iters)
        alpha = torch.zeros(num_iters)
        beta = torch.zeros(num_iters)

        b = b / torch.norm(b)
        Q[:, 0] = b
        u = b

        r = mv_closure(u)
        a = u.dot(r)
        b = r - a * u

        if b.sum() == 0:
            b = b + 1e-10

        beta[0] = torch.norm(b)
        alpha[0] = a

        for k in range(1, num_iters):
            u, b, alpha_k, beta_k = self._lanczos_step(u, b, mv_closure, Q[:, :k])

            if b.sum() == 0:
                b = b + 1e-10

            alpha[k] = alpha_k
            beta[k] = beta_k
            Q[:, k] = u

            if math.fabs(beta[k]) < 1e-5:
                break

        alpha = alpha[:k + 1]
        beta = beta[1:k + 1]
        Q = Q[:, :k + 1]
        T = torch.diag(alpha) + torch.diag(beta, 1) + torch.diag(beta, -1)
        return Q, T

    def _lanczos_step(self, u, v, mv_closure, Q):
        norm_v = torch.norm(v)
        orig_u = u

        u = v / norm_v

        if Q.size()[1] == 1:
            u = u - Q.mul((Q.t().mv(u)).expand_as(Q))
        else:
            u = u - Q.mv(Q.t().mv(u))

        u = u / torch.norm(u)

        r = mv_closure(u) - norm_v * orig_u

        a = u.dot(r)
        v = r - a * u
        return u, v, a, norm_v

    def evaluate(self, A, n, funcs):
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
        if isinstance(A, torch.Tensor) and A.numel() == 1:
            return math.fabs(A.squeeze()[0])

        if isinstance(A, torch.Tensor):
            def mv_closure(v):
                return A.mv(v)
        else:
            mv_closure = A

        V = torch.sign(torch.randn(n, self.num_random_probes))
        V.div_(torch.norm(V, 2, 0).expand_as(V))

        results = [0] * len(funcs)

        for j in range(self.num_random_probes):
            vj = V[:, j]
            Q, T = self.lanczos(mv_closure, vj)

            # Eigendecomposition of a Tridiagonal matrix
            # O(n^2) time/convergence with QR iteration,
            # or O(n log n) with fast multipole (TODO).
            [f, Y] = T.symeig(eigenvectors=True)
            for i, func in enumerate(funcs):
                results[i] = results[i] + n / float(self.num_random_probes) * (Y[0, :].pow(2).dot(func(f)))

        return results