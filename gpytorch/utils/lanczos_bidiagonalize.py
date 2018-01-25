import torch


class LanczosBidiagonalize(object):
    def __init__(self, cls=None, max_iter=20):
        self.cls = cls or torch.Tensor
        self.max_iter = max_iter

    def _reorthogonalize(self, matrix, vector, tol=1e-5):
        """
        Ensure that the given vector is orthogonal to all columns of the given matrix, up to a specified absolute inner
        product tolerance.
        """
        norms = matrix.t().matmul(vector)
        while any(torch.abs(norms) > tol):
            vector = vector - torch.sum(norms * matrix, dim=1)
            vector = vector / torch.norm(vector)
            norms = matrix.t().matmul(vector)

        return vector

    def lanczos_bidiagonalize(self, matmul_closure, initial_vector):
        matrix_size = len(initial_vector)
        # For now just do num_iters iterations
        num_iters = self.max_iter

        initial_vector = initial_vector / torch.norm(initial_vector)
        Q = self.cls(matrix_size, num_iters).zero_()
        P = self.cls(matrix_size, num_iters).zero_()
        beta = self.cls(num_iters).zero_()
        alpha = self.cls(num_iters).zero_()

        Q[:, 0] = initial_vector
        for i in range(num_iters):
            if i == 0:
                P[:, i] = matmul_closure(Q[:, 0])
            else:
                P[:, i] = matmul_closure(Q[:, i]) - beta[i - 1]*P[:, i - 1]
                P[:, i] = P[:, i] - torch.sum(P[:, :i].t().matmul(P[:, i]) * P[:, :i], dim=1)

            alpha[i] = torch.norm(P[:, i], 2)
            P[:, i] = P[:, i] / alpha[i]

            if i > 1:
                P[:, i] = self._reorthogonalize(P[:, :i], P[:, i])

            if i < num_iters - 1:
                # Compute next right vector
                Q[:, i + 1] = matmul_closure(P[:, i])

                # Reorthogonalize
                norms = Q[:, :i + 1].t().matmul(Q[:, i + 1])
                Q[:, i + 1] = Q[:, i + 1] - torch.sum(norms * Q[:, :i + 1], dim=1)

                # Normalize right vector
                beta[i] = torch.norm(Q[:, i + 1], 2)
                Q[:, i + 1] = Q[:, i + 1] / beta[i]

                Q[:, i + 1] = self._reorthogonalize(Q[:, :i + 1], Q[:, i + 1])

        B = self.cls(num_iters, num_iters).zero_()
        B = torch.diag(alpha) + torch.diag(beta[:-1], 1)

        return P, B, Q
