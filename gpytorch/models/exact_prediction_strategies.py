#!/usr/bin/env python3

import functools
import string

import torch

from .. import settings
from ..lazy import (
    AddedDiagLazyTensor,
    BatchRepeatLazyTensor,
    ConstantMulLazyTensor,
    InterpolatedLazyTensor,
    LazyEvaluatedKernelTensor,
    LowRankRootAddedDiagLazyTensor,
    MatmulLazyTensor,
    NonLazyTensor,
    RootLazyTensor,
    ZeroLazyTensor,
    delazify,
    lazify,
)
from ..utils.cholesky import psd_safe_cholesky
from ..utils.interpolation import left_interp, left_t_interp
from ..utils.memoize import add_to_cache, cached, clear_cache_hook, pop_from_cache


def prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood):
    train_train_covar = train_prior_dist.lazy_covariance_matrix
    if isinstance(train_train_covar, LazyEvaluatedKernelTensor):
        cls = train_train_covar.kernel.prediction_strategy
    else:
        cls = DefaultPredictionStrategy
    return cls(train_inputs, train_prior_dist, train_labels, likelihood)


class DefaultPredictionStrategy(object):
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, root=None, inv_root=None):
        # Get training shape
        self._train_shape = train_prior_dist.event_shape

        # Flatten the training labels
        train_labels = train_labels.reshape(*train_labels.shape[: -len(self.train_shape)], self._train_shape.numel())

        self.train_inputs = train_inputs
        self.train_prior_dist = train_prior_dist
        self.train_labels = train_labels
        self.likelihood = likelihood
        self._last_test_train_covar = None
        mvn = self.likelihood(train_prior_dist, train_inputs)
        self.lik_train_train_covar = mvn.lazy_covariance_matrix

        if root is not None:
            add_to_cache(self.lik_train_train_covar, "root_decomposition", RootLazyTensor(root))

        if inv_root is not None:
            add_to_cache(self.lik_train_train_covar, "root_inv_decomposition", RootLazyTensor(inv_root))

    def __deepcopy__(self, memo):
        # deepcopying prediction strategies of a model evaluated on inputs that require gradients fails
        # with RuntimeError (Only Tensors created explicitly by the user (graph leaves) support the deepcopy
        # protocol at the moment). Overwriting this method make sure that the prediction strategies of a
        # model are set to None upon deepcopying.
        pass

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        """
        Computes a cache for K_X*X (K_XX + sigma^2 I)^-1 K_X*X if possible. By default, this does no work and returns
        the first argument.

        Args:
            train_train_covar_inv_root (:obj:`torch.tensor`): a root of (K_XX + sigma^2 I)^-1
            test_train_covar (:obj:`torch.tensor`): the observed noise (from the likelihood)

        Returns
            - A precomputed cache
        """
        res = train_train_covar_inv_root
        if settings.detach_test_caches.on():
            res = res.detach()

        if res.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            res.grad_fn.register_hook(wrapper)

        return res

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        r"""
        Computes :math:`K_{X^{*}X} S` given a precomputed cache
        Where :math:`S` is a tensor such that :math:`SS^{\top} = (K_{XX} + \sigma^2 I)^{-1}`

        Args:
            precomputed_cache (:obj:`torch.tensor`): What was computed in _exact_predictive_covar_inv_quad_form_cache
            test_train_covar (:obj:`torch.tensor`): The observed noise (from the likelihood)

        Returns
            :obj:`~gpytorch.lazy.LazyTensor`: :math:`K_{X^{*}X} S`
        """
        # Here the precomputed cache represents S,
        # where S S^T = (K_XX + sigma^2 I)^-1
        return test_train_covar.matmul(precomputed_cache)

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        """
        Returns a new PredictionStrategy that incorporates the specified inputs and targets as new training data.

        This method is primary responsible for updating the mean and covariance caches. To add fantasy data to a
        GP model, use the :meth:`~gpytorch.models.ExactGP.get_fantasy_model` method.

        Args:
            - :attr:`inputs` (Tensor `b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`): Locations of fantasy
                observations.
            - :attr:`targets` (Tensor `b1 x ... x bk x m` or `f x b1 x ... x bk x m`): Labels of fantasy observations.
            - :attr:`full_inputs` (Tensor `b1 x ... x bk x n+m x d` or `f x b1 x ... x bk x n+m x d`): Training data
                concatenated with fantasy inputs
            - :attr:`full_targets` (Tensor `b1 x ... x bk x n+m` or `f x b1 x ... x bk x n+m`): Training labels
                concatenated with fantasy labels.
            - :attr:`full_output` (:class:`gpytorch.distributions.MultivariateNormal`): Prior called on full_inputs

        Returns:
            - :class:`DefaultPredictionStrategy`
                A `DefaultPredictionStrategy` model with `n + m` training examples, where the `m` fantasy examples have
                been added and all test-time caches have been updated.
        """
        full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

        batch_shape = full_inputs[0].shape[:-2]

        full_mean = full_mean.view(*batch_shape, -1)
        num_train = self.num_train

        # Evaluate fant x train and fant x fant covariance matrices, leave train x train unevaluated.
        fant_fant_covar = full_covar[..., num_train:, num_train:]
        fant_mean = full_mean[..., num_train:]
        mvn = self.train_prior_dist.__class__(fant_mean, fant_fant_covar)
        fant_likelihood = self.likelihood.get_fantasy_likelihood(**kwargs)
        mvn_obs = fant_likelihood(mvn, inputs, **kwargs)

        fant_fant_covar = mvn_obs.covariance_matrix
        fant_train_covar = delazify(full_covar[..., num_train:, :num_train])

        self.fantasy_inputs = inputs
        self.fantasy_targets = targets

        r"""
        Compute a new mean cache given the old mean cache.

        We have \alpha = K^{-1}y, and we want to solve [K U; U' S][a; b] = [y; y_f], where U' is fant_train_covar,
        S is fant_fant_covar, and y_f is (targets - fant_mean)

        To do this, we solve the bordered linear system of equations for [a; b]:
            AQ = U  # Q = fant_solve
            [S - U'Q]b = y_f - U'\alpha   ==> b = [S - U'Q]^{-1}(y_f - U'\alpha)
            a = \alpha - Qb
        """
        # Get cached K inverse decomp. (or compute if we somehow don't already have the covariance cache)
        K_inverse = self.lik_train_train_covar.root_inv_decomposition()
        fant_solve = K_inverse.matmul(fant_train_covar.transpose(-2, -1))

        # Solve for "b", the lower portion of the *new* \\alpha corresponding to the fantasy points.
        schur_complement = fant_fant_covar - fant_train_covar.matmul(fant_solve)

        # we'd like to use a less hacky approach for the following, but einsum can be much faster than
        # than unsqueezing/squeezing here (esp. in backward passes), unfortunately it currenlty has some
        # issues with broadcasting: https://github.com/pytorch/pytorch/issues/15671
        prefix = string.ascii_lowercase[: max(fant_train_covar.dim() - self.mean_cache.dim() - 1, 0)]
        ftcm = torch.einsum(prefix + "...yz,...z->" + prefix + "...y", [fant_train_covar, self.mean_cache])

        small_system_rhs = targets - fant_mean - ftcm
        small_system_rhs = small_system_rhs.unsqueeze(-1)
        # Schur complement of a spd matrix is guaranteed to be positive definite
        schur_cholesky = psd_safe_cholesky(schur_complement)
        fant_cache_lower = torch.cholesky_solve(small_system_rhs, schur_cholesky)

        # Get "a", the new upper portion of the cache corresponding to the old training points.
        fant_cache_upper = self.mean_cache.unsqueeze(-1) - fant_solve.matmul(fant_cache_lower)

        fant_cache_upper = fant_cache_upper.squeeze(-1)
        fant_cache_lower = fant_cache_lower.squeeze(-1)

        # New mean cache.
        fant_mean_cache = torch.cat((fant_cache_upper, fant_cache_lower), dim=-1)

        # now update the root and root inverse
        new_lt = self.lik_train_train_covar.cat_rows(fant_train_covar, fant_fant_covar)
        new_root = new_lt.root_decomposition().root.evaluate()
        new_covar_cache = new_lt.root_inv_decomposition().root.evaluate()

        # Expand inputs accordingly if necessary (for fantasies at the same points)
        if full_inputs[0].dim() <= full_targets.dim():
            fant_batch_shape = full_targets.shape[:1]
            n_batch = len(full_mean.shape[:-1])
            repeat_shape = fant_batch_shape + torch.Size([1] * n_batch)
            full_inputs = [fi.expand(fant_batch_shape + fi.shape) for fi in full_inputs]
            full_mean = full_mean.expand(fant_batch_shape + full_mean.shape)
            full_covar = BatchRepeatLazyTensor(full_covar, repeat_shape)
            new_root = BatchRepeatLazyTensor(NonLazyTensor(new_root), repeat_shape)
            # no need to repeat the covar cache, broadcasting will do the right thing

        # Create new DefaultPredictionStrategy object
        fant_strat = self.__class__(
            train_inputs=full_inputs,
            train_prior_dist=self.train_prior_dist.__class__(full_mean, full_covar),
            train_labels=full_targets,
            likelihood=fant_likelihood,
            root=new_root,
            inv_root=new_covar_cache,
        )
        add_to_cache(fant_strat, "mean_cache", fant_mean_cache)
        add_to_cache(fant_strat, "covar_cache", new_covar_cache)
        return fant_strat

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        train_train_covar = self.lik_train_train_covar
        train_train_covar_inv_root = delazify(train_train_covar.root_inv_decomposition().root)
        return self._exact_predictive_covar_inv_quad_form_cache(train_train_covar_inv_root, self._last_test_train_covar)

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean, train_train_covar = mvn.loc, mvn.lazy_covariance_matrix

        train_labels_offset = (self.train_labels - train_mean).unsqueeze(-1)
        mean_cache = train_train_covar.evaluate_kernel().inv_matmul(train_labels_offset).squeeze(-1)

        if settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()

        if mean_cache.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            mean_cache.grad_fn.register_hook(wrapper)

        return mean_cache

    @property
    def num_train(self):
        return self._train_shape.numel()

    @property
    def train_shape(self):
        return self._train_shape

    def exact_prediction(self, joint_mean, joint_covar):
        # Find the components of the distribution that contain test data
        test_mean = joint_mean[..., self.num_train :]
        # For efficiency - we can make things more efficient
        if joint_covar.size(-1) <= settings.max_eager_kernel_size.value():
            test_covar = joint_covar[..., self.num_train :, :].evaluate()
            test_test_covar = test_covar[..., self.num_train :]
            test_train_covar = test_covar[..., : self.num_train]
        else:
            test_test_covar = joint_covar[..., self.num_train :, self.num_train :]
            test_train_covar = joint_covar[..., self.num_train :, : self.num_train]

        return (
            self.exact_predictive_mean(test_mean, test_train_covar),
            self.exact_predictive_covar(test_test_covar, test_train_covar),
        )

    def exact_predictive_mean(self, test_mean, test_train_covar):
        """
        Computes the posterior predictive covariance of a GP

        Args:
            test_mean (:obj:`torch.tensor`): The test prior mean
            test_train_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test and train inputs

        Returns:
            :obj:`torch.tensor`: The predictive posterior mean of the test points
        """
        # NOTE TO FUTURE SELF:
        # You **cannot* use addmv here, because test_train_covar may not actually be a non lazy tensor even for an exact
        # GP, and using addmv requires you to delazify test_train_covar, which is obviously a huge no-no!
        res = (test_train_covar @ self.mean_cache.unsqueeze(-1)).squeeze(-1)
        res = res + test_mean

        return res

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        """
        Computes the posterior predictive covariance of a GP

        Args:
            test_train_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test and train inputs
            test_test_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test inputs

        Returns:
            :obj:`gpytorch.lazy.LazyTensor`: A LazyTensor representing the predictive posterior covariance of the
                                               test points
        """
        if settings.fast_pred_var.on():
            self._last_test_train_covar = test_train_covar

        if settings.skip_posterior_variances.on():
            return ZeroLazyTensor(*test_test_covar.size())

        if settings.fast_pred_var.off():
            dist = self.train_prior_dist.__class__(
                torch.zeros_like(self.train_prior_dist.mean), self.train_prior_dist.lazy_covariance_matrix
            )
            if settings.detach_test_caches.on():
                train_train_covar = self.likelihood(dist, self.train_inputs).lazy_covariance_matrix.detach()
            else:
                train_train_covar = self.likelihood(dist, self.train_inputs).lazy_covariance_matrix

            test_train_covar = delazify(test_train_covar)
            train_test_covar = test_train_covar.transpose(-1, -2)
            covar_correction_rhs = train_train_covar.inv_matmul(train_test_covar)
            # For efficiency
            if torch.is_tensor(test_test_covar):
                # We can use addmm in the 2d case
                if test_test_covar.dim() == 2:
                    return lazify(
                        torch.addmm(test_test_covar, test_train_covar, covar_correction_rhs, beta=1, alpha=-1)
                    )
                else:
                    return lazify(test_test_covar + test_train_covar @ covar_correction_rhs.mul(-1))
            # In other cases - we'll use the standard infrastructure
            else:
                return test_test_covar + MatmulLazyTensor(test_train_covar, covar_correction_rhs.mul(-1))

        precomputed_cache = self.covar_cache
        covar_inv_quad_form_root = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache, test_train_covar)
        if torch.is_tensor(test_test_covar):
            return lazify(
                torch.add(
                    test_test_covar, covar_inv_quad_form_root @ covar_inv_quad_form_root.transpose(-1, -2), alpha=-1
                )
            )
        else:
            return test_test_covar + MatmulLazyTensor(
                covar_inv_quad_form_root, covar_inv_quad_form_root.transpose(-1, -2).mul(-1)
            )


class InterpolatedPredictionStrategy(DefaultPredictionStrategy):
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, uses_wiski=False):
        train_prior_dist = train_prior_dist.__class__(
            train_prior_dist.mean, train_prior_dist.lazy_covariance_matrix.evaluate_kernel()
        )
        super().__init__(train_inputs, train_prior_dist, train_labels, likelihood)
        # covar = self.train_prior_dist.lazy_covariance_matrix.evaluate_kernel()
        # if isinstance(covar, LazyEvaluatedKernelTensor):
        #     covar = covar.evaluate_kernel()
        # self.train_prior_dist = self.train_prior_dist.__class__(
        #     self.train_prior_dist.mean, covar
        # )
        self.uses_wiski = uses_wiski

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        train_interp_indices = test_train_covar.right_interp_indices
        train_interp_values = test_train_covar.right_interp_values
        base_lazy_tensor = test_train_covar.base_lazy_tensor
        base_size = base_lazy_tensor.size(-1)
        res = base_lazy_tensor.matmul(
            left_t_interp(train_interp_indices, train_interp_values, train_train_covar_inv_root, base_size)
        )
        return res

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        # Here the precomputed cache represents K_UU W S,
        # where S S^T = (K_XX + sigma^2 I)^-1
        test_interp_indices = test_train_covar.left_interp_indices
        test_interp_values = test_train_covar.left_interp_values
        res = left_interp(test_interp_indices, test_interp_values, precomputed_cache)
        return res

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        r"""
        Implements the fantasy strategy described in https://arxiv.org/abs/2103.01454.
        """
        full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

        batch_shape = full_inputs[0].shape[:-2]

        full_mean = full_mean.view(*batch_shape, -1)
        num_train = self.num_train

        # Evaluate fant x train and fant x fant covariance matrices, leave train x train unevaluated.
        fant_fant_covar = full_covar[..., num_train:, num_train:].evaluate_kernel()
        fant_mean = full_mean[..., num_train:]

        fant_wmat = self.prepare_dense_wmat(fant_fant_covar)

        fant_likelihood = self.likelihood.get_fantasy_likelihood(**kwargs)
        fant_noise = fant_likelihood.noise_covar(fant_wmat.transpose(-1, -2) if len(fant_wmat.shape) > 2 else fant_wmat)
        fant_root_vector = fant_noise.sqrt_inv_matmul(fant_wmat.transpose(-1, -2)).transpose(-1, -2)

        new_wmat = self.interp_inner_prod.add_low_rank(fant_root_vector.evaluate())
        mean_diff = (targets - fant_mean).unsqueeze(-1)
        new_interp_response_cache = self.interp_response_cache + fant_wmat.matmul(fant_noise.inv_matmul(mean_diff))

        # Create new DefaultPredictionStrategy object
        fant_strat = self.__class__(
            train_inputs=full_inputs,
            train_prior_dist=self.train_prior_dist.__class__(full_mean, full_covar),
            train_labels=full_targets,
            likelihood=fant_likelihood,
            uses_wiski=True,
        )
        add_to_cache(fant_strat, "interp_inner_prod", new_wmat)
        add_to_cache(fant_strat, "interp_response_cache", new_interp_response_cache)
        return fant_strat

    def prepare_dense_wmat(self, covar=None):
        # prepare the w matrix which is batch shape x m x n, where n = covar.shape[-2]
        if covar is None:
            covar = self.train_prior_dist.lazy_covariance_matrix
        wmat = covar._sparse_left_interp_t(covar.left_interp_indices, covar.left_interp_values).to_dense()
        return lazify(wmat)

    @property
    @cached(name="interp_inner_prod")
    def interp_inner_prod(self):
        # the W'W cache
        wmat = self.prepare_dense_wmat()
        noise_term = self.likelihood.noise_covar(wmat.transpose(-1, -2) if len(wmat.shape) > 2 else wmat)
        interp_inner_prod = wmat.matmul(noise_term.inv_matmul(wmat.transpose(-1, -2)))
        return interp_inner_prod

    @property
    @cached(name="interp_response_cache")
    def interp_response_cache(self):
        wmat = self.prepare_dense_wmat()
        noise_term = self.likelihood.noise_covar(wmat.transpose(-1, -2) if len(wmat.shape) > 2 else wmat)
        demeaned_train_targets = self.train_labels - self.train_prior_dist.mean
        dinv_y = noise_term.inv_matmul(demeaned_train_targets.unsqueeze(-1))
        return wmat.matmul(dinv_y)

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        train_train_covar = self.train_prior_dist.lazy_covariance_matrix
        train_interp_indices = train_train_covar.left_interp_indices
        train_interp_values = train_train_covar.left_interp_values

        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean, train_train_covar_with_noise = mvn.mean, mvn.lazy_covariance_matrix

        mean_diff = (self.train_labels - train_mean).unsqueeze(-1)
        train_train_covar_inv_labels = train_train_covar_with_noise.inv_matmul(mean_diff)

        # New root factor
        base_size = train_train_covar.base_lazy_tensor.size(-1)
        mean_cache = train_train_covar.base_lazy_tensor.matmul(
            left_t_interp(train_interp_indices, train_interp_values, train_train_covar_inv_labels, base_size)
        )

        # Prevent backprop through this variable
        if settings.detach_test_caches.on():
            return mean_cache.detach()
        else:
            return mean_cache

    @property
    @cached(name="fantasy_mean_cache")
    def fantasy_mean_cache(self):
        # first construct K_UU
        train_train_covar = self.train_prior_dist.lazy_covariance_matrix
        inducing_covar = train_train_covar.base_lazy_tensor

        # now get L such that LL' \approx WD^{-1}W'
        interp_inner_prod_root = self.interp_inner_prod.root_decomposition(method="cholesky").root
        # M = KL
        inducing_compression_matrix = inducing_covar.matmul(interp_inner_prod_root)

        # Q = L'KL + 1
        current_qmatrix = interp_inner_prod_root.transpose(-1, -2).matmul(inducing_compression_matrix).add_jitter(1.0)

        # m = K_UU WD^{-1}(y - \mu)
        inducing_covar_response = inducing_covar.matmul(self.interp_response_cache)

        # L' m
        root_space_projection = interp_inner_prod_root.transpose(-1, -2).matmul(inducing_covar_response)
        # Q^{-1} (L' m)
        qmat_solve = current_qmatrix.inv_matmul(root_space_projection)

        mean_cache = inducing_covar_response - inducing_compression_matrix @ qmat_solve

        # Prevent backprop through this variable
        if settings.detach_test_caches.on():
            return mean_cache.detach()
        else:
            return mean_cache

    @property
    @cached(name="fantasy_covar_cache")
    def fantasy_covar_cache(self):
        train_train_covar = self.train_prior_dist.lazy_covariance_matrix
        inducing_covar = train_train_covar.base_lazy_tensor

        # we need to enforce a cholesky here for numerical stability
        interp_inner_prod_root = self.interp_inner_prod.root_decomposition(method="cholesky").root
        inducing_compression_matrix = inducing_covar.matmul(interp_inner_prod_root)

        current_qmatrix = interp_inner_prod_root.transpose(-1, -2).matmul(inducing_compression_matrix).add_jitter(1.0)

        if settings.fast_pred_var.on():
            qmat_inv_root = current_qmatrix.root_inv_decomposition()
            # to lazify you have to evaluate the inverse root which is slow
            # otherwise, you can't backprop your way through it
            inner_cache = RootLazyTensor(inducing_compression_matrix.matmul(qmat_inv_root.root.evaluate()))
        else:
            inner_cache = inducing_compression_matrix.matmul(
                current_qmatrix.inv_matmul(inducing_compression_matrix.transpose(-1, -2))
            )

        # Precomputed factor
        if settings.fast_pred_samples.on():
            predictive_covar_cache = inducing_covar - inner_cache
            inside_root = predictive_covar_cache.root_decomposition(method="cholesky").root
            # Prevent backprop through this variable
            if settings.detach_test_caches.on():
                inside_root = inside_root.detach()
            covar_cache = inside_root, None
        else:
            root = inner_cache.root_decomposition(method="cholesky").root

            # Prevent backprop through this variable
            if settings.detach_test_caches.on():
                root = root.detach()
            covar_cache = None, root

        return covar_cache

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        # Get inverse root
        train_train_covar = self.train_prior_dist.lazy_covariance_matrix
        train_interp_indices = train_train_covar.left_interp_indices
        train_interp_values = train_train_covar.left_interp_values

        # Get probe vectors for inverse root
        num_probe_vectors = settings.fast_pred_var.num_probe_vectors()
        num_inducing = train_train_covar.base_lazy_tensor.size(-1)
        vector_indices = torch.randperm(num_inducing).type_as(train_interp_indices)
        probe_vector_indices = vector_indices[:num_probe_vectors]
        test_vector_indices = vector_indices[num_probe_vectors : 2 * num_probe_vectors]

        probe_interp_indices = probe_vector_indices.unsqueeze(1)
        probe_test_interp_indices = test_vector_indices.unsqueeze(1)
        dtype = train_train_covar.dtype
        device = train_train_covar.device
        probe_interp_values = torch.ones(num_probe_vectors, 1, dtype=dtype, device=device)

        batch_shape = train_train_covar.base_lazy_tensor.batch_shape
        probe_vectors = InterpolatedLazyTensor(
            train_train_covar.base_lazy_tensor,
            train_interp_indices.expand(*batch_shape, *train_interp_indices.shape[-2:]),
            train_interp_values.expand(*batch_shape, *train_interp_values.shape[-2:]),
            probe_interp_indices.expand(*batch_shape, *probe_interp_indices.shape[-2:]),
            probe_interp_values.expand(*batch_shape, *probe_interp_values.shape[-2:]),
        ).evaluate()
        test_vectors = InterpolatedLazyTensor(
            train_train_covar.base_lazy_tensor,
            train_interp_indices.expand(*batch_shape, *train_interp_indices.shape[-2:]),
            train_interp_values.expand(*batch_shape, *train_interp_values.shape[-2:]),
            probe_test_interp_indices.expand(*batch_shape, *probe_test_interp_indices.shape[-2:]),
            probe_interp_values.expand(*batch_shape, *probe_interp_values.shape[-2:]),
        ).evaluate()

        # Put data through the likelihood
        dist = self.train_prior_dist.__class__(
            torch.zeros_like(self.train_prior_dist.mean), self.train_prior_dist.lazy_covariance_matrix
        )
        train_train_covar_plus_noise = self.likelihood(dist, self.train_inputs).lazy_covariance_matrix

        # Get inverse root
        train_train_covar_inv_root = train_train_covar_plus_noise.root_inv_decomposition(
            initial_vectors=probe_vectors, test_vectors=test_vectors
        ).root
        train_train_covar_inv_root = train_train_covar_inv_root.evaluate()

        # New root factor
        root = self._exact_predictive_covar_inv_quad_form_cache(train_train_covar_inv_root, self._last_test_train_covar)

        # Precomputed factor
        if settings.fast_pred_samples.on():
            inside = train_train_covar.base_lazy_tensor + RootLazyTensor(root).mul(-1)
            inside_root = inside.root_decomposition().root.evaluate()
            # Prevent backprop through this variable
            if settings.detach_test_caches.on():
                inside_root = inside_root.detach()
            covar_cache = inside_root, None
        else:
            # Prevent backprop through this variable
            if settings.detach_test_caches.on():
                root = root.detach()
            covar_cache = None, root

        return covar_cache

    def exact_prediction(self, joint_mean, joint_covar):
        # Find the components of the distribution that contain test data
        test_mean = joint_mean[..., self.num_train :]
        test_test_covar = joint_covar[..., self.num_train :, self.num_train :].evaluate_kernel()
        test_train_covar = joint_covar[..., self.num_train :, : self.num_train].evaluate_kernel()

        return (
            self.exact_predictive_mean(test_mean, test_train_covar),
            self.exact_predictive_covar(test_test_covar, test_train_covar),
        )

    def exact_predictive_mean(self, test_mean, test_train_covar):
        precomputed_cache = self.fantasy_mean_cache if self.uses_wiski else self.mean_cache
        test_interp_indices = test_train_covar.left_interp_indices
        test_interp_values = test_train_covar.left_interp_values
        res = left_interp(test_interp_indices, test_interp_values, precomputed_cache).squeeze(-1) + test_mean
        return res

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        if settings.fast_pred_var.off() and settings.fast_pred_samples.off():
            return super(InterpolatedPredictionStrategy, self).exact_predictive_covar(test_test_covar, test_train_covar)

        self._last_test_train_covar = test_train_covar
        test_interp_indices = test_train_covar.left_interp_indices
        test_interp_values = test_train_covar.left_interp_values

        if self.uses_wiski:
            precomputed_cache = self.fantasy_covar_cache
            fps = settings.fast_pred_samples.on()
            if fps:
                root = left_interp(test_interp_indices, test_interp_values, precomputed_cache[0].evaluate())
                res = RootLazyTensor(root)
            else:
                root = left_interp(test_interp_indices, test_interp_values, precomputed_cache[1].evaluate())
                res = test_test_covar + RootLazyTensor(root).mul(-1)
            return res
        else:
            precomputed_cache = self.covar_cache
            fps = settings.fast_pred_samples.on()
            if (fps and precomputed_cache[0] is None) or (not fps and precomputed_cache[1] is None):
                pop_from_cache(self, "covar_cache")
                precomputed_cache = self.covar_cache

            # Compute the exact predictive posterior
            if settings.fast_pred_samples.on():
                res = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache[0], test_train_covar)
                res = RootLazyTensor(res)
            else:
                root = left_interp(test_interp_indices, test_interp_values, precomputed_cache[1])
                res = test_test_covar + RootLazyTensor(root).mul(-1)
            return res


class RFFPredictionStrategy(DefaultPredictionStrategy):
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood):
        super().__init__(train_inputs, train_prior_dist, train_labels, likelihood)
        self.train_prior_dist = self.train_prior_dist.__class__(
            self.train_prior_dist.mean, self.train_prior_dist.lazy_covariance_matrix.evaluate_kernel()
        )

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        raise NotImplementedError("Fantasy observation updates not yet supported for models using RFFs")

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        lt = self.train_prior_dist.lazy_covariance_matrix
        if isinstance(lt, ConstantMulLazyTensor):
            constant = lt.expanded_constant
            lt = lt.base_lazy_tensor
        else:
            constant = torch.tensor(1.0, dtype=lt.dtype, device=lt.device)

        train_factor = lt.root.evaluate()
        train_train_covar = self.lik_train_train_covar
        inner_term = (
            torch.eye(train_factor.size(-1), dtype=train_factor.dtype, device=train_factor.device)
            - (train_factor.transpose(-1, -2) @ train_train_covar.inv_matmul(train_factor)) * constant
        )
        return psd_safe_cholesky(inner_term, jitter=settings.cholesky_jitter.value())

    def exact_prediction(self, joint_mean, joint_covar):
        # Find the components of the distribution that contain test data
        test_mean = joint_mean[..., self.num_train :]
        test_test_covar = joint_covar[..., self.num_train :, self.num_train :].evaluate_kernel()
        test_train_covar = joint_covar[..., self.num_train :, : self.num_train].evaluate_kernel()

        return (
            self.exact_predictive_mean(test_mean, test_train_covar),
            self.exact_predictive_covar(test_test_covar, test_train_covar),
        )

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        if settings.skip_posterior_variances.on():
            return ZeroLazyTensor(*test_test_covar.size())

        if isinstance(test_test_covar, ConstantMulLazyTensor):
            constant = test_test_covar.expanded_constant
            test_test_covar = test_test_covar.base_lazy_tensor
        else:
            constant = torch.tensor(1.0, dtype=test_test_covar.dtype, device=test_test_covar.device)

        covar_cache = self.covar_cache
        factor = test_test_covar.root.evaluate() * constant.sqrt()
        res = RootLazyTensor(factor @ covar_cache)
        return res


class SGPRPredictionStrategy(DefaultPredictionStrategy):
    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        # Here, the covar_cache is going to be K_{UU}^{-1/2} K_{UX}( K_{XX} + \sigma^2 I )^{-1} K_{XU} K_{UU}^{-1/2}
        # This is easily computed using Woodbury
        # K_{XX} + \sigma^2 I = R R^T + \sigma^2 I
        #                     = \sigma^{-2} ( I - \sigma^{-2} R (I + \sigma^{-2} R^T R)^{-1} R^T  )
        train_train_covar = self.lik_train_train_covar.evaluate_kernel()

        # Get terms needed for woodbury
        root = train_train_covar._lazy_tensor.root_decomposition().root.evaluate()  # R
        inv_diag = train_train_covar._diag_tensor.inverse()  # \sigma^{-2}

        # Form LT using woodbury
        ones = torch.tensor(1.0, dtype=root.dtype, device=root.device)
        chol_factor = lazify(root.transpose(-1, -2) @ (inv_diag @ root)).add_diag(ones)  # (I + \sigma^{-2} R^T R)^{-1}
        woodbury_term = inv_diag @ torch.triangular_solve(
            root.transpose(-1, -2), chol_factor.cholesky().evaluate(), upper=False
        )[0].transpose(-1, -2)
        # woodbury_term @ woodbury_term^T = \sigma^{-2} R (I + \sigma^{-2} R^T R)^{-1} R^T \sigma^{-2}

        inverse = AddedDiagLazyTensor(inv_diag, MatmulLazyTensor(-woodbury_term, woodbury_term.transpose(-1, -2)))
        # \sigma^{-2} ( I - \sigma^{-2} R (I + \sigma^{-2} R^T R)^{-1} R^T  )

        return root.transpose(-1, -2) @ (inverse @ root)

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        raise NotImplementedError(
            "Fantasy observation updates not yet supported for models using SGPRPredictionStrategy"
        )

    def exact_prediction(self, joint_mean, joint_covar):
        from ..kernels.inducing_point_kernel import InducingPointKernel

        # Find the components of the distribution that contain test data
        test_mean = joint_mean[..., self.num_train :]

        # If we're in lazy evaluation mode, let's use the base kernel of the SGPR output to compute the prior covar
        test_test_covar = joint_covar[..., self.num_train :, self.num_train :]
        if isinstance(test_test_covar, LazyEvaluatedKernelTensor) and isinstance(
            test_test_covar.kernel, InducingPointKernel
        ):
            test_test_covar = LazyEvaluatedKernelTensor(
                test_test_covar.x1,
                test_test_covar.x2,
                test_test_covar.kernel.base_kernel,
                test_test_covar.last_dim_is_batch,
                **test_test_covar.params,
            )

        test_train_covar = joint_covar[..., self.num_train :, : self.num_train].evaluate_kernel()

        return (
            self.exact_predictive_mean(test_mean, test_train_covar),
            self.exact_predictive_covar(test_test_covar, test_train_covar),
        )

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        covar_cache = self.covar_cache
        # covar_cache = K_{UU}^{-1/2} K_{UX}( K_{XX} + \sigma^2 I )^{-1} K_{XU} K_{UU}^{-1/2}

        # Decompose test_train_covar = l, r
        # Main case: test_x and train_x are different - test_train_covar is a MatmulLazyTensor
        if isinstance(test_train_covar, MatmulLazyTensor):
            L = test_train_covar.left_lazy_tensor.evaluate()
        # Edge case: test_x and train_x are the same - test_train_covar is a LowRankRootAddedDiagLazyTensor
        elif isinstance(test_train_covar, LowRankRootAddedDiagLazyTensor):
            L = test_train_covar._lazy_tensor.root.evaluate()
        else:
            # We should not hit this point of the code - this is to catch potential bugs in GPyTorch
            raise ValueError(
                "Expected SGPR output to be a MatmulLazyTensor or AddedDiagLazyTensor. "
                f"Got {test_train_covar.__class__.__name__} instead. "
                "This is likely a bug in GPyTorch."
            )

        res = test_test_covar - (L @ (covar_cache @ L.transpose(-1, -2)))
        return res
