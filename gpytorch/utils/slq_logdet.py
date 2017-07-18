import torch
import math


class SLQLogDet(object):
    """
    Implements an approximate log determinant calculation for symmetric positive definite matrices
    using stochastic Lanczos quadrature as described in Ubaru et al., 2016 here:

                                http://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf
    """
    def __init__(self, max_iter=15, num_random_probes=10):
        self.max_iter = max_iter
        self.num_random_probes = num_random_probes

    def lanczos(self, A, b):
        """
        Performs self.max_iter (at most n) iterations of the Lanczos iteration to decompose A as:
            AQ = QT
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

        r = A.mv(u)
        a = u.dot(r)
        b = r - a * u

        if b.sum() == 0:
            b = b + 1e-10

        beta[0] = torch.norm(b)
        alpha[0] = a

        for k in range(1, num_iters):
            u, b, alpha_k, beta_k = self._lanczos_step(u, b, A, Q[:, :k])

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

    def _lanczos_step(self, u, v, B, Q):
        norm_v = torch.norm(v)
        orig_u = u

        u = v / norm_v

        if Q.size()[1] == 1:
            u = u - Q.mul((Q.t().mv(u)).expand_as(Q))
        else:
            u = u - Q.mv(Q.t().mv(u))

        u = u / torch.norm(u)

        r = B.mv(u) - norm_v * orig_u

        a = u.dot(r)
        v = r - a * u
        return u, v, a, norm_v

    def logdet(self, A):
        if A.numel() == 1:
            return math.fabs(A.squeeze()[0])

        n = len(A)
        jitter = torch.eye(n) * 1e-5
        A = A + jitter
        V = torch.sign(torch.randn(n, self.num_random_probes))
        V.div_(torch.norm(V, 2, 0).expand_as(V))

        ld = 0

        for j in range(self.num_random_probes):
            vj = V[:, j]
            Q, T = self.lanczos(A, vj)

            # Eigendecomposition of a Tridiagonal matrix
            # O(n^2) time/convergence with QR iteration,
            # or O(n log n) with fast multipole (TODO).
            [f, Y] = T.symeig(eigenvectors=True)
            ld = ld + n / float(self.num_random_probes) * (Y[0, :].pow(2).dot(f.log_()))

        return ld
