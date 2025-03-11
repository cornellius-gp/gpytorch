#!/usr/bin/env python3

from typing import Any, Optional

import torch
from jaxtyping import Float
from linear_operator import to_dense
from linear_operator.operators import DiagLinearOperator, LinearOperator, TriangularLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import LongTensor, Tensor

from ..distributions import MultivariateNormal
from ..models import ApproximateGP, ExactGP
from ..module import Module
from ..utils.errors import CachingError
from ..utils.memoize import add_to_cache, cached, pop_from_cache
from ..utils.nearest_neighbors import NNUtil
from ._variational_distribution import _VariationalDistribution
from .mean_field_variational_distribution import MeanFieldVariationalDistribution
from .unwhitened_variational_strategy import UnwhitenedVariationalStrategy


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

        The current implementation only supports :obj:`~gpytorch.variational.MeanFieldVariationalDistribution`.

        We recommend installing the `faiss`_ library (requiring separate package installment)
        for nearest neighbor search, which is significantly faster than the `scikit-learn` nearest neighbor search.
        GPyTorch will automatically use `faiss` if it is installed, but will revert to `scikit-learn` otherwise.

        Different inducing point orderings will produce in different nearest neighbor approximations.


    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param k: Number of nearest neighbors.
    :param training_batch_size: The number of data points that will be in the training batch size.
    :param jitter_val: Amount of diagonal jitter to add for covariance matrix numerical stability.
    :param compute_full_kl: Whether to compute full kl divergence or stochastic estimate.

    .. _Wu et al (2022):
        https://arxiv.org/pdf/2202.01694.pdf
    .. _VNNGP tutorial:
        examples/04_Variational_and_Approximate_GPs/VNNGP.html
    .. _faiss:
        https://github.com/facebookresearch/faiss
    """

    def __init__(
        self,
        model: ApproximateGP,
        inducing_points: Float[Tensor, "... M D"],
        variational_distribution: Float[_VariationalDistribution, "... M"],
        k: int,
        training_batch_size: Optional[int] = None,
        jitter_val: Optional[float] = 1e-3,
        compute_full_kl: Optional[bool] = False,
    ):
        assert isinstance(
            variational_distribution, MeanFieldVariationalDistribution
        ), "Currently, NNVariationalStrategy only supports MeanFieldVariationalDistribution."

        super().__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations=False, jitter_val=jitter_val
        )

        # Model
        object.__setattr__(self, "model", model)

        self.inducing_points = inducing_points
        self.M, self.D = inducing_points.shape[-2:]
        self.k = k
        assert self.k < self.M, (
            f"Number of nearest neighbors k must be smaller than the number of inducing points, "
            f"but got k = {k}, M = {self.M}."
        )

        self._inducing_batch_shape: torch.Size = inducing_points.shape[:-2]
        self._model_batch_shape: torch.Size = self._variational_distribution.variational_mean.shape[:-1]
        self._batch_shape: torch.Size = torch.broadcast_shapes(self._inducing_batch_shape, self._model_batch_shape)

        self.nn_util: NNUtil = NNUtil(
            k, dim=self.D, batch_shape=self._inducing_batch_shape, device=inducing_points.device
        )
        self._compute_nn()
        # otherwise, no nearest neighbor approximation is used

        self.training_batch_size = training_batch_size if training_batch_size is not None else self.M
        self._set_training_iterator()

        self.compute_full_kl = compute_full_kl

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> Float[MultivariateNormal, "... M"]:
        out = self.model.forward(self.inducing_points)
        res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter(self.jitter_val))
        return res

    def _cholesky_factor(
        self, induc_induc_covar: Float[LinearOperator, "... M M"]
    ) -> Float[TriangularLinearOperator, "... M M"]:
        # Uncached version
        L = psd_safe_cholesky(to_dense(induc_induc_covar))
        return TriangularLinearOperator(L)

    def __call__(
        self, x: Float[Tensor, "... N D"], prior: bool = False, **kwargs: Any
    ) -> Float[MultivariateNormal, "... N"]:
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x, **kwargs)

        if x is not None:
            # Make sure x and inducing points have the same batch shape
            if not (self.inducing_points.shape[:-2] == x.shape[:-2]):
                try:
                    x = x.expand(*self.inducing_points.shape[:-2], *x.shape[-2:]).contiguous()
                except RuntimeError:
                    raise RuntimeError(
                        f"x batch shape must match or broadcast with the inducing points' batch shape, "
                        f"but got x batch shape = {x.shape[:-2]}, "
                        f"inducing points batch shape = {self.inducing_points.shape[:-2]}."
                    )

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()

            # (Maybe) initialize variational distribution
            if not self.variational_params_initialized.item():
                prior_dist = self.prior_distribution
                self._variational_distribution.variational_mean.data.copy_(prior_dist.mean)
                self._variational_distribution.variational_mean.data.add_(
                    torch.randn_like(prior_dist.mean), alpha=self._variational_distribution.mean_init_std
                )
                # initialize with a small variational stddev for quicker conv. of kl divergence
                self._variational_distribution._variational_stddev.data.copy_(torch.tensor(1e-2))
                self.variational_params_initialized.fill_(1)

            return self.forward(
                x, self.inducing_points, inducing_values=None, variational_inducing_covar=None, **kwargs
            )
        else:
            # Ensure inducing_points and x are the same size
            inducing_points = self.inducing_points
            return self.forward(x, inducing_points, inducing_values=None, variational_inducing_covar=None, **kwargs)

    def forward(
        self,
        x: Float[Tensor, "... N D"],
        inducing_points: Float[Tensor, "... M D"],
        inducing_values: Float[Tensor, "... M"],
        variational_inducing_covar: Optional[Float[LinearOperator, "... M M"]] = None,
        **kwargs: Any,
    ) -> Float[MultivariateNormal, "... N"]:
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
                expanded_variational_mean = self._variational_distribution.variational_mean.expand(
                    *self._batch_shape, self.M
                )
                expanded_variational_var = (
                    self._variational_distribution._variational_stddev.expand(*self._batch_shape, self.M) ** 2
                )

                predictive_mean = expanded_variational_mean.gather(-1, expanded_x_indices)
                predictive_var = expanded_variational_var.gather(-1, expanded_x_indices)

                # sample a different indices for stochastic estimation of kl
                kl_indices = self._get_training_indices()

            kl = self._kl_divergence(kl_indices)
            add_to_cache(self, "kl_divergence_memo", kl)

            return MultivariateNormal(predictive_mean, DiagLinearOperator(predictive_var))
        else:
            nn_indices = self.nn_util.find_nn_idx(x.float())

            x_batch_shape = x.shape[:-2]
            batch_shape = torch.broadcast_shapes(self._batch_shape, x_batch_shape)
            x_bsz = x.shape[-2]
            assert nn_indices.shape == (*x_batch_shape, x_bsz, self.k), nn_indices.shape

            # select K nearest neighbors from inducing points for test point x
            expanded_nn_indices = nn_indices.unsqueeze(-1).expand(*x_batch_shape, x_bsz, self.k, self.D)
            expanded_inducing_points = inducing_points.unsqueeze(-2).expand(*x_batch_shape, self.M, self.k, self.D)
            inducing_points = expanded_inducing_points.gather(-3, expanded_nn_indices)
            assert inducing_points.shape == (*x_batch_shape, x_bsz, self.k, self.D)

            # get variational mean and covar for nearest neighbors
            inducing_values = self._variational_distribution.variational_mean
            expanded_inducing_values = inducing_values.unsqueeze(-1).expand(*batch_shape, self.M, self.k)
            expanded_nn_indices = nn_indices.expand(*batch_shape, x_bsz, self.k)
            inducing_values = expanded_inducing_values.gather(-2, expanded_nn_indices)
            assert inducing_values.shape == (*batch_shape, x_bsz, self.k)

            variational_stddev = self._variational_distribution._variational_stddev
            assert variational_stddev.shape == (*self._model_batch_shape, self.M)
            expanded_variational_stddev = variational_stddev.unsqueeze(-1).expand(*batch_shape, self.M, self.k)
            variational_inducing_covar = expanded_variational_stddev.gather(-2, expanded_nn_indices) ** 2
            assert variational_inducing_covar.shape == (*batch_shape, x_bsz, self.k)
            variational_inducing_covar = DiagLinearOperator(variational_inducing_covar)
            assert variational_inducing_covar.shape == (*batch_shape, x_bsz, self.k, self.k)

            # Make everything batch mode
            x = x.unsqueeze(-2)
            assert x.shape == (*x_batch_shape, x_bsz, 1, self.D)
            x = x.expand(*batch_shape, x_bsz, 1, self.D)

            # Compute forward mode in the standard way
            _batch_dims = tuple(range(len(batch_shape)))
            _x = x.permute((-3,) + _batch_dims + (-2, -1))  # (x_bsz, *batch_shape, 1, D)

            # inducing_points.shape (*x_batch_shape, x_bsz, self.k, self.D)
            inducing_points = inducing_points.expand(*batch_shape, x_bsz, self.k, self.D)
            _inducing_points = inducing_points.permute((-3,) + _batch_dims + (-2, -1))  # (x_bsz, *batch_shape, k, D)
            _inducing_values = inducing_values.permute((-2,) + _batch_dims + (-1,))
            _variational_inducing_covar = variational_inducing_covar.permute((-3,) + _batch_dims + (-2, -1))
            dist = super().forward(_x, _inducing_points, _inducing_values, _variational_inducing_covar, **kwargs)

            _x_batch_dims = tuple(range(1, 1 + len(batch_shape)))
            predictive_mean = dist.mean  # (x_bsz, *x_batch_shape, 1)
            predictive_covar = dist.covariance_matrix  # (x_bsz, *x_batch_shape, 1, 1)
            predictive_mean = predictive_mean.permute(_x_batch_dims + (0, -1))
            predictive_covar = predictive_covar.permute(_x_batch_dims + (0, -2, -1))

            # Undo batch mode
            predictive_mean = predictive_mean.squeeze(-1)
            predictive_var = predictive_covar.squeeze(-2).squeeze(-1)
            assert predictive_var.shape == predictive_covar.shape[:-2]
            assert predictive_mean.shape == predictive_covar.shape[:-2]

            # Return the distribution
            return MultivariateNormal(predictive_mean, DiagLinearOperator(predictive_var))

    def get_fantasy_model(
        self,
        inputs: Float[Tensor, "... N D"],
        targets: Float[Tensor, "... N"],
        mean_module: Optional[Module] = None,
        covar_module: Optional[Module] = None,
        **kwargs,
    ) -> ExactGP:
        raise NotImplementedError(
            f"No fantasy model support for {self.__class__.__name__}. "
            "Only VariationalStrategy and UnwhitenedVariationalStrategy are currently supported."
        )

    def _set_training_iterator(self) -> None:
        self._training_indices_iter = 0
        if self.training_batch_size == self.M:
            self._training_indices_iterator = (torch.arange(self.M, device=self.inducing_points.device),)
        else:
            # The first training batch always contains the first k inducing points
            # This is because computing the KL divergence for the first k inducing points is special-cased
            # (since the first k inducing points have < k neighbors)
            # Note that there is a special function _firstk_kl_helper for this
            training_indices = torch.randperm(self.M - self.k, device=self.inducing_points.device) + self.k
            self._training_indices_iterator = (torch.arange(self.k),) + training_indices.split(self.training_batch_size)
        self._total_training_batches = len(self._training_indices_iterator)

    def _get_training_indices(self) -> LongTensor:
        self.current_training_indices = self._training_indices_iterator[self._training_indices_iter]
        self._training_indices_iter += 1
        if self._training_indices_iter == self._total_training_batches:
            self._set_training_iterator()
        return self.current_training_indices

    def _firstk_kl_helper(self) -> Float[Tensor, "..."]:
        # Compute the KL divergence for first k inducing points
        train_x_firstk = self.inducing_points[..., : self.k, :]
        full_output = self.model.forward(train_x_firstk)

        induc_mean, induc_induc_covar = full_output.mean, full_output.lazy_covariance_matrix

        induc_induc_covar = induc_induc_covar.add_jitter(self.jitter_val)
        prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)

        inducing_values = self._variational_distribution.variational_mean[..., : self.k]
        variational_covar_fisrtk = self._variational_distribution._variational_stddev[..., : self.k] ** 2
        variational_inducing_covar = DiagLinearOperator(variational_covar_fisrtk)

        variational_distribution = MultivariateNormal(inducing_values, variational_inducing_covar)
        kl = torch.distributions.kl.kl_divergence(variational_distribution, prior_dist)  # model_batch_shape
        return kl

    def _stochastic_kl_helper(self, kl_indices: Float[Tensor, "n_batch"]) -> Float[Tensor, "..."]:  # noqa: F821
        # Compute the KL divergence for a mini batch of the rest M-k inducing points
        # See paper appendix for kl breakdown
        kl_bs = len(kl_indices)  # training_batch_size
        variational_mean = self._variational_distribution.variational_mean  # (*model_bs, M)
        variational_stddev = self._variational_distribution._variational_stddev

        # (1) compute logdet_q
        inducing_point_log_variational_covar = (variational_stddev[..., kl_indices] ** 2).log()
        logdet_q = torch.sum(inducing_point_log_variational_covar, dim=-1)  # model_bs

        # (2) compute lodet_p
        # Select a mini-batch of inducing points according to kl_indices
        inducing_points = self.inducing_points[..., kl_indices, :].expand(*self._batch_shape, kl_bs, self.D)
        # (*bs, kl_bs, D)
        # Select their K nearest neighbors
        nearest_neighbor_indices = self.nn_xinduce_idx[..., kl_indices - self.k, :].to(inducing_points.device)
        # (*bs, kl_bs, K)
        expanded_inducing_points_all = self.inducing_points.unsqueeze(-2).expand(
            *self._batch_shape, self.M, self.k, self.D
        )
        expanded_nearest_neighbor_indices = nearest_neighbor_indices.unsqueeze(-1).expand(
            *self._batch_shape, kl_bs, self.k, self.D
        )
        nearest_neighbors = expanded_inducing_points_all.gather(-3, expanded_nearest_neighbor_indices)
        # (*bs, kl_bs, K, D)

        # Compute prior distribution
        # Move the kl_bs dimension to the first dimension to enable batch covar_module computation
        nearest_neighbors_ = nearest_neighbors.permute((-3,) + tuple(range(len(self._batch_shape))) + (-2, -1))
        # (kl_bs, *bs, K, D)
        inducing_points_ = inducing_points.permute((-2,) + tuple(range(len(self._batch_shape))) + (-1,))
        # (kl_bs, *bs, D)
        full_output = self.model.forward(torch.cat([nearest_neighbors_, inducing_points_.unsqueeze(-2)], dim=-2))
        full_mean, full_covar = full_output.mean, full_output.covariance_matrix

        # Mean terms
        _undo_permute_dims = tuple(range(1, 1 + len(self._batch_shape))) + (0, -1)
        nearest_neighbors_prior_mean = full_mean[..., : self.k].permute(_undo_permute_dims)  # (*inducing_bs, kl_bs, K)
        inducing_prior_mean = full_mean[..., self.k :].permute(_undo_permute_dims).squeeze(-1)  # (*inducing_bs, kl_bs)
        # Covar terms
        nearest_neighbors_prior_cov = full_covar[..., : self.k, : self.k]
        nearest_neighbors_inducing_prior_cross_cov = full_covar[..., : self.k, self.k :]
        inducing_prior_cov = full_covar[..., self.k :, self.k :]
        inducing_prior_cov = (
            inducing_prior_cov.squeeze(-1).squeeze(-1).permute((-1,) + tuple(range(len(self._batch_shape))))
        )

        # Interpolation term K_nn^{-1} k_{nu}
        interp_term = torch.linalg.solve(
            nearest_neighbors_prior_cov + self.jitter_val * torch.eye(self.k, device=self.inducing_points.device),
            nearest_neighbors_inducing_prior_cross_cov,
        ).squeeze(
            -1
        )  # (kl_bs, *inducing_bs, K)
        interp_term = interp_term.permute(_undo_permute_dims)  # (*inducing_bs, kl_bs, K)
        nearest_neighbors_inducing_prior_cross_cov = nearest_neighbors_inducing_prior_cross_cov.squeeze(-1).permute(
            _undo_permute_dims
        )  # k_{n(j),j}, (*inducing_bs, kl_bs, K)

        invquad_term_for_F = torch.sum(
            interp_term * nearest_neighbors_inducing_prior_cross_cov, dim=-1
        )  # (*inducing_bs, kl_bs)

        inducing_prior_cov = self.model.covar_module.forward(
            inducing_points, inducing_points, diag=True
        )  # (*inducing_bs, kl_bs)

        F = inducing_prior_cov - invquad_term_for_F
        F = F + self.jitter_val
        # K_uu - k_un K_nn^{-1} k_nu
        logdet_p = F.log().sum(dim=-1)  # shape: inducing_bs

        # (3) compute trace_term
        expanded_variational_stddev = variational_stddev.unsqueeze(-1).expand(*self._batch_shape, self.M, self.k)
        expanded_variational_mean = variational_mean.unsqueeze(-1).expand(*self._batch_shape, self.M, self.k)
        expanded_nearest_neighbor_indices = nearest_neighbor_indices.expand(*self._batch_shape, kl_bs, self.k)
        nearest_neighbor_variational_covar = (
            expanded_variational_stddev.gather(-2, expanded_nearest_neighbor_indices) ** 2
        )  # (*batch_shape, kl_bs, k)
        bjsquared_s_nearest_neighbors = torch.sum(
            interp_term**2 * nearest_neighbor_variational_covar, dim=-1
        )  # (*batch_shape, kl_bs)
        inducing_point_variational_covar = variational_stddev[..., kl_indices] ** 2  # (model_bs, kl_bs)
        trace_term = (1.0 / F * (bjsquared_s_nearest_neighbors + inducing_point_variational_covar)).sum(
            dim=-1
        )  # batch_shape

        # (4) compute invquad_term
        nearest_neighbors_variational_mean = expanded_variational_mean.gather(-2, expanded_nearest_neighbor_indices)
        Bj_m_nearest_neighbors = torch.sum(
            interp_term * (nearest_neighbors_variational_mean - nearest_neighbors_prior_mean), dim=-1
        )
        inducing_variational_mean = variational_mean[..., kl_indices]
        invquad_term = torch.sum(
            (inducing_variational_mean - inducing_prior_mean - Bj_m_nearest_neighbors) ** 2 / F, dim=-1
        )

        kl = (logdet_p - logdet_q - kl_bs + trace_term + invquad_term) * (1.0 / 2)
        assert kl.shape == self._batch_shape, kl.shape

        return kl

    def _kl_divergence(
        self, kl_indices: Optional[LongTensor] = None, batch_size: Optional[int] = None
    ) -> Float[Tensor, "..."]:
        if self.compute_full_kl or (self._total_training_batches == 1):
            if batch_size is None:
                batch_size = self.training_batch_size
            kl = self._firstk_kl_helper()
            for kl_indices in torch.split(torch.arange(self.k, self.M), batch_size):
                kl += self._stochastic_kl_helper(kl_indices)
        else:
            # compute a stochastic estimate
            assert kl_indices is not None
            if self._training_indices_iter == 1:
                assert len(kl_indices) == self.k, (
                    f"kl_indices sould be the first batch data of length k, "
                    f"but got len(kl_indices) = {len(kl_indices)} and k = {self.k}."
                )
                kl = self._firstk_kl_helper() * self.M / self.k
            else:
                kl = self._stochastic_kl_helper(kl_indices) * self.M / len(kl_indices)
        return kl

    def kl_divergence(self) -> Float[Tensor, "..."]:
        try:
            return pop_from_cache(self, "kl_divergence_memo")
        except CachingError:
            raise RuntimeError("KL Divergence of variational strategy was called before nearest neighbors were set.")

    def _compute_nn(self) -> "NNVariationalStrategy":
        with torch.no_grad():
            inducing_points_fl = self.inducing_points.data.float()
            self.nn_util.set_nn_idx(inducing_points_fl)
            self.nn_xinduce_idx = self.nn_util.build_sequential_nn_idx(inducing_points_fl)
            #  shape (*_inducing_batch_shape, M-k, k)
        return self
