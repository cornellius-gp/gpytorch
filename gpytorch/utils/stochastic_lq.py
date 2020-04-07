#!/usr/bin/env python3

import torch

from .lanczos import lanczos_tridiag


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
            - cls - Tensor constructor - to ensure correct type (default - default tensor)
            - max_iter (scalar) - The number of Lanczos iterations to perform. Increasing this makes the estimate of
                tr(f(A)) more accurate in expectation -- that is, the average value returned has lower error.
            - num_random_probes (scalar) - The number of random probes to use in the stochastic trace estimation.
                Increasing this makes the estimate of tr(f(A)) lower variance -- that is, the value
                returned is more consistent.
        """
        self.max_iter = max_iter
        self.num_random_probes = num_random_probes

    def lanczos_batch(self, matmul_closure, rhs_vectors):
        return lanczos_tridiag(
            matmul_closure,
            self.max_iter,
            init_vecs=rhs_vectors,
            dtype=rhs_vectors.dtype,
            device=rhs_vectors.device,
            batch_shape=rhs_vectors.shape[-2:],
            matrix_shape=torch.Size((rhs_vectors.size(-2), rhs_vectors.size(-2))),
        )

    def evaluate(self, matrix_shape, eigenvalues, eigenvectors, funcs):
        r"""
        Computes tr(f(A)) for an arbitrary list of functions, where f(A) is equivalent to applying the function
        elementwise to the eigenvalues of A, i.e., if A = V\LambdaV^{T}, then f(A) = Vf(\Lambda)V^{T}, where
        f(\Lambda) is applied elementwise.
        Note that calling this function with a list of functions to apply is significantly more efficient than
        calling it multiple times with one function -- each additional function after the first requires negligible
        additional computation.

        Args:
            - matrix_shape (torch.Size()) - size of underlying matrix (not including batch dimensions)
            - eigenvalues (Tensor n_probes x ...batch_shape x k) - batches of eigenvalues from Lanczos tridiag mats
            - eigenvectors (Tensor n_probes x ...batch_shape x k x k) - batches of eigenvectors from " " "
            - funcs (list of closures) - A list of functions [f_1,...,f_k]. tr(f_i(A)) is computed for each function.
                Each function in the closure should expect to take a torch vector of eigenvalues as input and apply
                the function elementwise. For example, to compute logdet(A) = tr(log(A)), [lambda x: x.log()] would
                be a reasonable value of funcs.

        Returns:
            - results (list of scalars) - The trace of each supplied function applied to the matrix, e.g.,
                [tr(f_1(A)),tr(f_2(A)),...,tr(f_k(A))].
        """
        batch_shape = torch.Size(eigenvalues.shape[1:-1])
        results = [torch.zeros(batch_shape, dtype=eigenvalues.dtype, device=eigenvalues.device) for _ in funcs]
        num_random_probes = eigenvalues.size(0)
        for j in range(num_random_probes):
            # These are (num_batch x k) and (num_batch x k x k)
            eigenvalues_for_probe = eigenvalues[j]
            eigenvectors_for_probe = eigenvectors[j]
            for i, func in enumerate(funcs):
                # First component of eigenvecs is (num_batch x k)
                eigenvecs_first_component = eigenvectors_for_probe[..., 0, :]
                func_eigenvalues = func(eigenvalues_for_probe)

                dot_products = (eigenvecs_first_component.pow(2) * func_eigenvalues).sum(-1)
                results[i] = results[i] + matrix_shape[-1] / float(num_random_probes) * dot_products

        return results
