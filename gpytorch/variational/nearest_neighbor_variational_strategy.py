#!/usr/bin/env python3


import torch

from .unwhitened_variational_strategy import UnwhitenedVariationalStrategy
from ..distributions import MultivariateNormal
from ..lazy import (
    NonLazyTensor,
    DiagLazyTensor,
    delazify,
    TriangularLazyTensor
)
from ..utils.memoize import add_to_cache, pop_from_cache, cached, clear_cache_hook
from ..utils.errors import CachingError
from ..utils.cholesky import psd_safe_cholesky
from ..utils.nearest_neighbors import NNUtil
from .. import settings


class NNVariationalStrategy(UnwhitenedVariationalStrategy):
    r"""
    This strategy sets all inducing point locations to observed inputs,
    and employs a :math:`k`-nearest-neighbor approximation. It was introduced as the
    `Variational Nearest Neighbor Gaussian Processes (VNNGP)` in `Wu et al (2022)`_.
    See the `VNNGP tutorial`_ for an example.

    VNNGP assumes a k-nearest-neighbor generative process for inducing points :math:`\mathbf u`,
    :math:`\mathbf q(\mathbf u) = \prod_{j=1}^M q(u_j | \mathbf u_{n(j)})`
    where :math:`n(j)` denotes the indices of :math:`k` nearest neighbors for :math:`u_j` among
    :math:`u_1, \cdots, u_{j-1}`. For any test observation :math:`\mathbf f`,
    VNNGP makes predictive inference conditioned on its :math:`k` nearest inducing points
    :math:`\mathbf u_{n(f)}`, i.e. :math:`p(f|\mathbf u_{n(f)})`.

    VNNGP's objective factorizes over inducing points and observations, making stochastic optimization over both
    immediately available. After a one-time cost of computing the :math:`k`-nearest neighbor structure,
    the training and inference complexity is :math:`O(k^3)`.
    Since VNNGP uses observations as inducing points, it is a user choice to either (1)
    use the same mini-batch of inducing points and observations (recommended),
    or (2) use different mini-batches of inducing points and observations. See the `VNNGP tutorial`_ for
    implementation and comparison.


    .. note::

        Current implementation only supports :obj:`~gpytorch.variational._VariationalDistribution`.

        It is recommended that this strategy is used with `faiss`_ (requiring separate package installment)
        as the backend for nearest neighbor search, which will greatly speed up on large-scale datasets
        over the default backend `scikit-learn`.

        Different ording of inducing points will result in different nearest neighbor approximation.
        We recommend randomizing the ordering of inducing points (i.e. the training data) before feeding them
        into the strategy if there is no other prior knowledge.


    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param torch.Tensor inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :type learn_inducing_locations: `bool`, optional

    .. _Wu et al (2022):
        https://arxiv.org/pdf/2202.01694.pdf
    .. _VNNGP tutorial:
        examples/04_Variational_and_Approximate_GPs/VNNGP.html
    .. _faiss:
        https://github.com/facebookresearch/faiss
    """

    def __init__(self, model, inducing_points, variational_distribution, k, training_batch_size):

        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations=False)
        # Make sure we don't try to initialize variational parameters - because of minibatching
        self.variational_params_initialized.fill_(1)

        # Model
        object.__setattr__(self, "model", model)

        self.inducing_points = inducing_points

        self.M = inducing_points.shape[-2]
        self.D = inducing_points.shape[-1]
        self.k = k
        assert self.k <= self.M, \
            f"Number of nearest neighbors k must be smaller than or equal to number of inducing points, but got k = {k}, M = {self.M}."

        self.nn_util = NNUtil(k, dim=self.D, device=inducing_points.device)
        self.compute_nn()

        self.training_batch_size = training_batch_size
        self.set_training_iterator()

    def _clear_cache(self):
        clear_cache_hook(self)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        out = self.model.forward(self.inducing_points)
        jitter_val = settings.nn_jitter.value(self.inducing_points.dtype)
        res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter(jitter_val))
        return res

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        return self._variational_distribution()

    def _cholesky_factor(self, induc_induc_covar):
        # Uncached version
        L = psd_safe_cholesky(delazify(induc_induc_covar))
        return TriangularLazyTensor(L)

    def __call__(self, x, prior=False, **kwargs):
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x, **kwargs)

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()
            return self.forward(x, self.inducing_points, None, None)
        else:
            # Ensure inducing_points and x are the same size
            inducing_points = self.inducing_points
            if x is not None:
                if inducing_points.shape[:-2] != x.shape[:-2]:
                    x, inducing_points = self._expand_inputs(x, inducing_points)

            # Get p(u)/q(u)
            variational_dist_u = self.variational_distribution

            return self.forward(
                x,
                inducing_points,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                **kwargs,
            )

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        if self.training:
            # In training mode, note that the full inducing points set = full training dataset
            # Users have the option to choose input None or a tensor of training data for x
            # If x is None, will sample training data from inducing points
            # Otherwise, will find the indices of inducing points that are equal to x
            if x is None:
                x_indices = self._get_training_indices()
                kl_indices = x_indices
            else:
                x_indices = self.nn_util.find_nn_idx(x.float(), k=1).squeeze(-1)
                #assert torch.equal(self.inducing_points[x_indices], x)

                # sample x_indices for KL computation
                kl_indices = self._get_training_indices()

            predictive_mean = self._variational_distribution.variational_mean[x_indices]
            predictive_var = self._variational_distribution._variational_stddev[x_indices] ** 2
            kl = self._kl_divergence(kl_indices)
            add_to_cache(self, "kl_divergence_memo", kl)

            return MultivariateNormal(predictive_mean, DiagLazyTensor(predictive_var))
        else:
            nn_indices = self.nn_util.find_nn_idx(x.float())

            # Make everything batch mode
            x = x.unsqueeze(-2)
            inducing_points = inducing_points[..., nn_indices, :]
            inducing_values = inducing_values[..., nn_indices]
            if variational_inducing_covar is not None:
                variational_inducing_covar = NonLazyTensor(
                    variational_inducing_covar[..., nn_indices.unsqueeze(-1), nn_indices.unsqueeze(-2)]
                )

            # Compute forward mode in the standard way
            dist = super().forward(x, inducing_points, inducing_values, variational_inducing_covar, **kwargs)
            predictive_mean = dist.mean
            predictive_covar = dist.covariance_matrix  # Should be 1 x 1

            # Undo batch mode
            predictive_mean = predictive_mean.squeeze(-1)
            predictive_var = predictive_covar.squeeze(-2).squeeze(-1)
            assert predictive_var.shape == predictive_covar.shape[:-2]
            assert predictive_mean.shape == predictive_covar.shape[:-2]

            # Return the distribution
            return MultivariateNormal(predictive_mean, DiagLazyTensor(predictive_var))

    def set_training_iterator(self):
        self._training_indices_iter = 0
        training_indices = torch.randperm(self.M - self.k, device=self.inducing_points.device) + self.k
        self._training_indices_iterator = (torch.arange(self.k),) + training_indices.split(self.training_batch_size)
        self._total_training_batches = len(self._training_indices_iterator)

    def _get_training_indices(self):
        self.current_training_indices = self._training_indices_iterator[self._training_indices_iter]
        self._training_indices_iter += 1
        if self._training_indices_iter == self._total_training_batches:
            self.set_training_iterator()
        return self.current_training_indices

    def firstk_kl_helper(self):
        # Compute the KL divergence for first k inducing points
        train_x_firstk = self.inducing_points[:self.k]
        full_output = self.model.forward(train_x_firstk)
        induc_mean, induc_induc_covar = full_output.mean, full_output.lazy_covariance_matrix
        jitter_val = settings.nn_jitter.value(self.inducing_points.dtype)
        induc_induc_covar = induc_induc_covar.add_jitter(jitter_val)
        prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)

        inducing_values = self._variational_distribution.variational_mean[:self.k]
        variational_covar_fisrtk = self._variational_distribution._variational_stddev[:self.k]**2
        variational_inducing_covar = DiagLazyTensor(variational_covar_fisrtk)

        variational_distribution = MultivariateNormal(inducing_values, variational_inducing_covar)
        kl = torch.distributions.kl.kl_divergence(variational_distribution, prior_dist).mean(dim=-1)
        return kl

    def stochastic_kl_helper(self, kl_indices):
        # Compute the KL divergence for a mini batch of the rest M-1 inducing points
        jitter_val = settings.nn_jitter.value(self.inducing_points.dtype)

        current_kl_bs = len(kl_indices)

        variational_mean = self._variational_distribution.variational_mean
        variational_stddev = self._variational_distribution._variational_stddev
        selected_log_variational_covar = (variational_stddev[kl_indices]**2).log()
        logdet_q = selected_log_variational_covar.sum()

        selected_xinduce = self.inducing_points[kl_indices]  # (kl_bs, D)

        nn_idx_for_selected_xinduce = self.nn_xinduce_idx[kl_indices - self.k]  # (kl_bs, k)
        nn_for_selected_xinduce = self.inducing_points[nn_idx_for_selected_xinduce]  # (kl_bs, k, D)

        cov = self.model.covar_module.forward(nn_for_selected_xinduce, nn_for_selected_xinduce)  # (kl_bs, k, k)
        cross_cov = self.model.covar_module.forward(nn_for_selected_xinduce, selected_xinduce.unsqueeze(-2))  # (kl_bs, k, D) * (kl_bs, 1, D) -> (kl_bs, k, 1)
        interp_term = torch.linalg.solve(cov+jitter_val*torch.eye(self.k, device=self.inducing_points.device), cross_cov).squeeze(-1) # (kl_bs, k)

        invquad_term_for_F = torch.sum(interp_term * cross_cov.squeeze(-1), dim=-1)
        F = self.model.covar_module.forward(selected_xinduce, selected_xinduce, diag=True) - invquad_term_for_F # (kl_bs, )
        F = F + jitter_val
        logdet_p = F.log().sum()

        nn_selected_variational_covar =  variational_stddev[nn_idx_for_selected_xinduce]**2
        Bjsq_s = torch.sum(interp_term**2 * nn_selected_variational_covar, dim=-1)  # (kl_bs, )
        selected_variational_covar = variational_stddev[kl_indices]**2
        trace_term = (1./F * (Bjsq_s + selected_variational_covar)).sum()

        Bj_m = torch.sum(interp_term * variational_mean[nn_idx_for_selected_xinduce], dim=-1)  # (kl_bs, )
        invquad_term = torch.sum((variational_mean[kl_indices] - Bj_m)**2 / F)
        invquad_term = invquad_term

        kl = 1./2 * (logdet_p - logdet_q - current_kl_bs + trace_term + invquad_term)
        return kl

    def _kl_divergence(self, kl_indices=None, compute_full=False, batch_size=None):
        if compute_full:
            if batch_size is None:
                batch_size = self.training_batch_size
            kl = self.firstk_kl_helper()
            for kl_indices in torch.split(torch.arange(self.k, self.M), batch_size):
                kl += self.stochastic_kl_helper(kl_indices)
        else:
            assert kl_indices is not None
            if (self._training_indices_iter == 1) or (self.M == self.k):
                assert len(kl_indices) == self.k, \
                    f"kl_indices sould be the first batch data of length k, " \
                    f"but got len(kl_indices) = {len(kl_indices)} and k = {self.k}."
                kl = self.firstk_kl_helper() * self.M / self.k
            else:
                kl = self.stochastic_kl_helper(kl_indices) * self.M / len(kl_indices)
        return kl

    def kl_divergence(self):
        try:
            return pop_from_cache(self, "kl_divergence_memo")
        except CachingError:
            raise RuntimeError("KL Divergence of variational strategy was called before nearest neighbors were set.")

    def compute_nn(self):
        with torch.no_grad():
            self.nn_util.set_nn_idx(self.inducing_points.data.float())
            self.nn_xinduce_idx = self.nn_util.build_sequential_nn_idx(
                self.inducing_points.data.float())
        return self

