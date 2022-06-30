#!/usr/bin/env python3


import torch
from .mean_field_variational_distribution import MeanFieldVariationalDistribution
from .unwhitened_variational_strategy import UnwhitenedVariationalStrategy
from ..distributions import MultivariateNormal
from ..lazy import (
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
        
        assert isinstance(variational_distribution, MeanFieldVariationalDistribution), "Currently, NNVariationalStrategy only supports MeanFieldVariationalDistribution."


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

        self._inducing_batch_shape = inducing_points.shape[:-2]
        self._model_batch_shape = self._variational_distribution.variational_mean.shape[:-1]
        self._batch_shape = torch.broadcast_shapes(self._inducing_batch_shape, self._model_batch_shape)

        self.nn_util = NNUtil(k, dim=self.D, batch_shape=self._inducing_batch_shape, device=inducing_points.device)
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

        if x is not None:
            assert self.inducing_points.shape[:-2] == x.shape[:-2], \
                f"Train data batch shape must matches inducing points batch shape, " \
                f"but got train data batch shape = {x.shape[:-2]}, inducing points batch shape = {self.inducing_points.shape[:-2]} "

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


            return self.forward(x,inducing_points,inducing_values=None, variational_inducing_covar=None,**kwargs)

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        if self.training:
            # In training mode, note that the full inducing points set = full training dataset
            # Users have the option to choose input None or a tensor of training data for x
            # If x is None, will sample training data from inducing points
            # Otherwise, will find the indices of inducing points that are equal to x
            if x is None:
                x_indices = self._get_training_indices()
                kl_indices = x_indices

                predictive_mean = self._variational_distribution.variational_mean[..., x_indices]
                predictive_var = self._variational_distribution._variational_stddev[..., x_indices] ** 2

            else:
                # find the indices of inducing points that correspond to x
                x_indices = self.nn_util.find_nn_idx(x.float(), k=1).squeeze(-1)  # (*inducing_batch_shape, batch_size)

                expanded_x_indices = x_indices.expand(*self._batch_shape, x_indices.shape[-1])
                expanded_variational_mean = self._variational_distribution.variational_mean.expand(*self._batch_shape, self.M)
                expanded_variational_var = self._variational_distribution._variational_stddev.expand(*self._batch_shape, self.M)**2

                predictive_mean = expanded_variational_mean.gather(-1, expanded_x_indices)
                predictive_var = expanded_variational_var.gather(-1, expanded_x_indices)

                # sample a different indices for stochastic estimation of kl
                kl_indices = self._get_training_indices()

            kl = self._kl_divergence(kl_indices)
            add_to_cache(self, "kl_divergence_memo", kl)

            return MultivariateNormal(predictive_mean, DiagLazyTensor(predictive_var))
        else:

            nn_indices = self.nn_util.find_nn_idx(x.float())

            x_batch_shape = x.shape[:-2]
            x_bsz = x.shape[-2]
            assert nn_indices.shape == (*x_batch_shape, x_bsz, self.k), nn_indices.shape

            expanded_nn_indices = nn_indices.unsqueeze(-1).expand(*x_batch_shape, x_bsz, self.k, self.D)
            expanded_inducing_points = inducing_points.unsqueeze(-2).expand(*x_batch_shape, self.M, self.k, self.D)
            inducing_points = expanded_inducing_points.gather(-3, expanded_nn_indices)
            assert inducing_points.shape == (*x_batch_shape, x_bsz, self.k, self.D)

            # get variational mean and covar for nearest neighbors
            batch_shape = torch.broadcast_shapes(self._model_batch_shape, x_batch_shape)
            inducing_values = self._variational_distribution.variational_mean
            expanded_inducing_values = inducing_values.unsqueeze(-1).expand(*batch_shape, self.M, self.k)
            expanded_nn_indices = nn_indices.expand(*batch_shape, x_bsz, self.k)
            inducing_values = expanded_inducing_values.gather(-2, expanded_nn_indices)
            assert inducing_values.shape == (*batch_shape, x_bsz, self.k)

            variational_stddev = self._variational_distribution._variational_stddev
            assert variational_stddev.shape == (*self._model_batch_shape, self.M)
            expanded_variational_stddev = variational_stddev.unsqueeze(-1).expand(*batch_shape, self.M, self.k)
            variational_inducing_covar = expanded_variational_stddev.gather(-2, expanded_nn_indices)**2
            assert variational_inducing_covar.shape == (*batch_shape, x_bsz, self.k)
            variational_inducing_covar = DiagLazyTensor(variational_inducing_covar)
            assert variational_inducing_covar.shape == (*batch_shape, x_bsz, self.k, self.k)

            # Make everything batch mode
            x = x.unsqueeze(-2)
            assert x.shape == (*x_batch_shape, x_bsz, 1, self.D)

            # Compute forward mode in the standard way
            dist = super().forward(x, inducing_points, inducing_values, variational_inducing_covar, **kwargs)
            predictive_mean = dist.mean # (*batch_shape, x_bsz, 1)
            predictive_covar = dist.covariance_matrix  # (*batch_shape, x_bsz, 1, 1)

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
        train_x_firstk = self.inducing_points[..., :self.k, :] # (*inducing_batch_shape, k, D)
        full_output = self.model.forward(train_x_firstk)

        induc_mean, induc_induc_covar = full_output.mean, full_output.lazy_covariance_matrix
        # (*inducing_batch_shape, k), (*inducing_batch_shape, k, k)

        jitter_val = settings.nn_jitter.value(self.inducing_points.dtype)
        induc_induc_covar = induc_induc_covar.add_jitter(jitter_val)
        prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)

        inducing_values = self._variational_distribution.variational_mean[..., :self.k]  # (*model_batch_shape, k)
        variational_covar_fisrtk = self._variational_distribution._variational_stddev[...,:self.k] ** 2  # (*model_batch_shape, k)
        variational_inducing_covar = DiagLazyTensor(variational_covar_fisrtk)  # (*model_batch_shape, k, k)

        variational_distribution = MultivariateNormal(inducing_values, variational_inducing_covar)
        kl = torch.distributions.kl.kl_divergence(variational_distribution, prior_dist)  # model_batch_shape
        assert kl.shape == self._variational_distribution.variational_mean.shape[:-1], kl.shape
        return kl

    def stochastic_kl_helper(self, kl_indices):
        # Compute the KL divergence for a mini batch of the rest M-1 inducing points
        # See paper appendix for kl breakdown
        jitter_val = settings.nn_jitter.value(self.inducing_points.dtype)
        kl_bs = len(kl_indices)
        variational_mean = self._variational_distribution.variational_mean # (*model_batch_shape, M)
        variational_stddev = self._variational_distribution._variational_stddev # (*model_batch_shape, M)

        # compute logdet_q
        inducing_point_log_variational_covar = (variational_stddev[..., kl_indices] ** 2).log() # (*model_batch_shape, kl_bs)
        logdet_q = torch.sum(inducing_point_log_variational_covar, dim=-1) # model_batch_shape

        # Select a mini-batch of inducing points according to kl_indices, and their k-nearest neighbors
        inducing_points = self.inducing_points[..., kl_indices, :]  # (*inducing_batch_shape, kl_bs, D)
        nearest_neighbor_indices = self.nn_xinduce_idx[..., kl_indices - self.k, :].to(inducing_points.device)  # (*inducing_batch_shape, kl_bs, k)
        expanded_inducing_points_all = self.inducing_points.unsqueeze(-2).expand(*self._inducing_batch_shape, self.M, self.k, self.D)
        expanded_nearest_neighbor_indices = nearest_neighbor_indices.unsqueeze(-1).expand(*self._inducing_batch_shape, kl_bs, self.k, self.D)
        nearest_neighbors = expanded_inducing_points_all.gather(-3, expanded_nearest_neighbor_indices)  # (*inducing_batch_shape, kl_bs, k, D)

        # compute interp_term
        cov = self.model.covar_module.forward(nearest_neighbors, nearest_neighbors)  # (*inducing-batch_shape, kl_bs, k, k)
        cross_cov = self.model.covar_module.forward(nearest_neighbors, inducing_points.unsqueeze(-2))  # (*induicng_batch_shape, kl_bs, k, 1)
        interp_term = torch.linalg.solve(cov+jitter_val*torch.eye(self.k, device=self.inducing_points.device), cross_cov).squeeze(-1) # (*inducing_batch_shape, kl_bs, k)

        # compte logdet_p
        invquad_term_for_F = torch.sum(interp_term * cross_cov.squeeze(-1), dim=-1)  #(*inducing_batch_shape, kl_bs)
        cov_inducing_points = self.model.covar_module.forward(inducing_points, inducing_points, diag=True)
        F = cov_inducing_points - invquad_term_for_F # (*inducing_batch_shape, kl_bs)
        F = F + jitter_val
        logdet_p = F.log().sum(dim=-1)

        # compute trace_term
        expanded_variational_stddev = variational_stddev.unsqueeze(-1).expand(*self._batch_shape, self.M, self.k)
        expanded_variational_mean = variational_mean.unsqueeze(-1).expand(*self._batch_shape, self.M, self.k)
        expanded_nearest_neighbor_indices = nearest_neighbor_indices.expand(*self._batch_shape, kl_bs, self.k)
        nearest_neighbor_variational_covar = expanded_variational_stddev.gather(-2, expanded_nearest_neighbor_indices)**2  # (*batch_shape, kl_bs, k)
        bjsquared_s = torch.sum(interp_term**2 * nearest_neighbor_variational_covar, dim=-1)  # (*batch_shape, kl_bs)
        inducing_point_covar = variational_stddev[..., kl_indices] ** 2
        trace_term = (1./F * (bjsquared_s + inducing_point_covar)).sum(dim=-1)

        # compute invquad_term
        nearest_neighbor_variational_mean = expanded_variational_mean.gather(-2, expanded_nearest_neighbor_indices) # (*batch_shape, kl_bs, k)
        Bj_m = torch.sum(interp_term * nearest_neighbor_variational_mean, dim=-1) # (*batch_shape, kl_bs)
        inducing_point_variational_mean = variational_mean[..., kl_indices] ** 2
        invquad_term = torch.sum((inducing_point_variational_mean - Bj_m)**2 / F, dim=-1)

        kl = 1./2 * (logdet_p - logdet_q - kl_bs + trace_term + invquad_term)
        assert kl.shape == self._batch_shape, kl.shape
        kl = kl.mean()

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
            inducing_points_fl = self.inducing_points.data.float()
            self.nn_util.set_nn_idx(inducing_points_fl)
            self.nn_xinduce_idx = self.nn_util.build_sequential_nn_idx(inducing_points_fl)
        return self

