import torch
from .lanczos import lanczos_tridiag


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
        return lanczos_tridiag(matmul_closure, self.max_iter, init_vecs=rhs_vectors, tensor_cls=self.cls)

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
                for k in range(Ts.size(1)):
                    T = Ts[j, k, :, :]
                    [f, Y] = T.symeig(eigenvectors=True)
                    for i, func in enumerate(funcs):
                        results[i][k] = results[i][k] + n / float(self.num_random_probes) * \
                            (Y[0, :].pow(2).dot(func(f + 1.1e-4)))

            else:
                T = Ts[j, :, :]
                [f, Y] = T.symeig(eigenvectors=True)
                for i, func in enumerate(funcs):
                    results[i] = results[i] + n / float(self.num_random_probes) * (Y[0, :].pow(2).dot(func(f + 1.1e-4)))

        return results
