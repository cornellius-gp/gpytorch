from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

    def evaluate(self, t_mats, matrix_size, eigenvalues, eigenvectors, funcs):
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
        if t_mats.dim() < 4:
            raise RuntimeError(
                "StochasticLQ expects t_mat to be (num_probe_vecs x num_batch x k x k), "
                "but received a Tensor with only {} dimensions. Try unsqueezing to add "
                "appropriate singleton dimensions if necessary.".format(t_mats.dim())
            )

        batch_size = t_mats.size(1)
        results = [t_mats.new(batch_size).zero_()] * len(funcs)
        num_random_probes = t_mats.size(0)
        for j in range(num_random_probes):
            # These are (num_batch x k) and (num_batch x k x k)
            eigenvalues_for_probe = eigenvalues[j]
            eigenvectors_for_probe = eigenvectors[j]
            for i, func in enumerate(funcs):
                # First component of eigenvecs is (num_batch x k)
                eigenvecs_first_component = eigenvectors_for_probe[:, 0, :]
                func_eigenvalues = func(eigenvalues_for_probe)

                dot_products = (eigenvecs_first_component.pow(2) * func_eigenvalues).sum(1)
                results[i] = results[i] + matrix_size / float(num_random_probes) * dot_products

        return results
