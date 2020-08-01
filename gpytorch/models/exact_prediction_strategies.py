#!/usr/bin/env python3

import functools
import string

import torch

from .. import settings
from ..lazy import (
    BatchRepeatLazyTensor,
    ConstantMulLazyTensor,
    InterpolatedLazyTensor,
    LazyEvaluatedKernelTensor,
    MatmulLazyTensor,
    NonLazyTensor,
    RootLazyTensor,
    SumLazyTensor,
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
        # Flatten the training labels
        train_shape = train_prior_dist.event_shape
        train_labels = train_labels.view(*train_labels.shape[: -len(train_shape)], train_shape.numel())

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
        schur_cholesky = psd_safe_cholesky(schur_complement, jitter=settings.cholesky_jitter.value())
        fant_cache_lower = torch.cholesky_solve(small_system_rhs, schur_cholesky)

        # Get "a", the new upper portion of the cache corresponding to the old training points.
        fant_cache_upper = self.mean_cache.unsqueeze(-1) - fant_solve.matmul(fant_cache_lower)

        fant_cache_upper = fant_cache_upper.squeeze(-1)
        fant_cache_lower = fant_cache_lower.squeeze(-1)

        # New mean cache.
        fant_mean_cache = torch.cat((fant_cache_upper, fant_cache_lower), dim=-1)

        """
        Compute a new covariance cache given the old covariance cache.

        We have access to K \\approx LL' and K^{-1} \\approx R^{-1}R^{-T}, where L and R are low rank matrices
        resulting from Lanczos (see the LOVE paper).

        To update R^{-1}, we first update L:
            [K U; U' S] = [L 0; A B][L' A'; 0 B']
        Solving this matrix equation, we get:
            K = LL' ==>       L = L
            U = LA' ==>       A = UR^{-1}
            S = AA' + BB' ==> B = cholesky(S - AA')

        Once we've computed Z = [L 0; A B], we have that the new kernel matrix [K U; U' S] \approx ZZ'. Therefore,
        we can form a pseudo-inverse of Z directly to approximate [K U; U' S]^{-1/2}.
        """
        # [K U; U' S] = [L 0; lower_left schur_root]
        batch_shape = fant_train_covar.shape[:-2]

        L_inverse = self.covar_cache
        L = self.lik_train_train_covar.root_decomposition().root
        m, n = L.shape[-2:]

        lower_left = fant_train_covar.matmul(L_inverse)
        schur = fant_fant_covar - lower_left.matmul(lower_left.transpose(-2, -1))
        schur_root = psd_safe_cholesky(schur, jitter=settings.cholesky_jitter.value())

        # Form new root Z = [L 0; lower_left schur_root]

        # # TODO: Special case triangular case once #1102 goes in
        # if isinstance(L, TriangularLazyTensor):
        #     # The whole thing is triangular, we can just do two triangular solves
        #     ...
        # else:

        L = delazify(L)
        num_fant = schur_root.size(-2)
        new_root = torch.zeros(*batch_shape, m + num_fant, n + num_fant, device=L.device, dtype=L.dtype)
        new_root[..., :m, :n] = L
        new_root[..., m:, :n] = lower_left
        new_root[..., m:, n:] = schur_root

        # Use pseudo-inverse of Z as new inv root

        if new_root.shape[-1] <= 2048:
            # Dispatch to CPU so long as pytorch/pytorch#22573 is not fixed
            device = new_root.device
            Q, R = torch.qr(new_root.cpu())
            Q = Q.to(device)
            R = R.to(device)
        else:
            Q, R = torch.qr(new_root)

        Rdiag = torch.diagonal(R, dim1=-2, dim2=-1)
        # if R is almost singular, add jitter
        zeroish = Rdiag.abs() < 1e-6
        if torch.any(zeroish):
            # can't use in-place operation here b/c it would mess up backward pass
            # haven't found a more elegant way to add a jitter diagonal yet...
            jitter_diag = 1e-6 * torch.sign(Rdiag) * zeroish.to(Rdiag)
            R = R + torch.diag_embed(jitter_diag)
        new_covar_cache = torch.triangular_solve(Q.transpose(-2, -1), R)[0].transpose(-2, -1)

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
        mean_cache = train_train_covar.inv_matmul(train_labels_offset).squeeze(-1)

        if settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()

        if mean_cache.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            mean_cache.grad_fn.register_hook(wrapper)

        return mean_cache

    @property
    def num_train(self):
        return self.train_prior_dist.event_shape.numel()

    @property
    def train_shape(self):
        return self.train_prior_dist.event_shape

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
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood):
        super().__init__(train_inputs, train_prior_dist, train_labels, likelihood)
        self.train_prior_dist = self.train_prior_dist.__class__(
            self.train_prior_dist.mean, self.train_prior_dist.lazy_covariance_matrix.evaluate_kernel()
        )

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
        raise NotImplementedError(
            "Fantasy observation updates not yet supported for models using InterpolatedLazyTensors"
        )

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
            probe_vectors, test_vectors
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
        precomputed_cache = self.mean_cache
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


class SumPredictionStrategy(DefaultPredictionStrategy):
    @property
    def _sub_strategies(self):
        sub_strategies = []
        for lazy_tensor in self.train_prior_dist.lazy_covariance_matrix.evaluate_kernel().lazy_tensors:
            pred_strat = prediction_strategy(
                self.train_inputs,
                self.train_prior_dist.__class__(self.train_prior_dist.mean, lazy_tensor),
                self.train_labels,
                self.likelihood,
            )
            sub_strategies.append(pred_strat)

        return sub_strategies

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        test_train_covar = lazify(test_train_covar).evaluate_kernel()
        if not isinstance(test_train_covar, SumLazyTensor):
            return super(SumPredictionStrategy, self)._exact_predictive_covar_inv_quad_form_cache(
                train_train_covar_inv_root, test_train_covar
            )
        else:
            return tuple(
                sub_strat._exact_predictive_covar_inv_quad_form_cache(train_train_covar_inv_root, test_train_covar_comp)
                for sub_strat, test_train_covar_comp in zip(self._sub_strategies, test_train_covar.lazy_tensors)
            )

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        # Here the precomputed cache is a list
        # where each component in the list is the precomputed cache for each component lazy tensor
        test_train_covar = lazify(test_train_covar).evaluate_kernel()
        if not isinstance(test_train_covar, SumLazyTensor):
            return super(SumPredictionStrategy, self)._exact_predictive_covar_inv_quad_form_root(
                precomputed_cache, test_train_covar
            )
        else:
            return sum(
                sub_strat._exact_predictive_covar_inv_quad_form_root(cache_comp, test_train_covar_comp)
                for sub_strat, cache_comp, test_train_covar_comp in zip(
                    self._sub_strategies, precomputed_cache, test_train_covar.evaluate_kernel().lazy_tensors
                )
            )


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
        return psd_safe_cholesky(inner_term)

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
