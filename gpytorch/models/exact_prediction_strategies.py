#!/usr/bin/env python3

import torch

from .. import settings
from ..distributions import MultivariateNormal
from ..lazy import (
    InterpolatedLazyTensor, LazyTensor, MatmulLazyTensor, RootLazyTensor, SumLazyTensor, ZeroLazyTensor,
    lazify, delazify
)
from ..utils.interpolation import left_interp, left_t_interp
from ..utils.memoize import cached
from ..utils.cholesky import cholesky_solve


_PREDICTION_STRATEGY_REGISTRY = {}


def register_prediction_strategy(lazy_tsr_type):
    if not isinstance(lazy_tsr_type, type) and issubclass(lazy_tsr_type, LazyTensor):
        raise TypeError("register_prediction_strategy expects a LazyTensor subtype but got {}".format(lazy_tsr_type))

    def decorator(cls):
        _PREDICTION_STRATEGY_REGISTRY[lazy_tsr_type] = cls
        return cls

    return decorator


def prediction_strategy(
    num_train, train_inputs, train_mean, train_train_covar, train_labels, likelihood, non_batch_train=False
):
    cls = _PREDICTION_STRATEGY_REGISTRY.get(type(train_train_covar), DefaultPredictionStrategy)
    return cls(num_train, train_inputs, train_mean, train_train_covar, train_labels, likelihood, non_batch_train)


class DefaultPredictionStrategy(object):
    def __init__(
        self, num_train, train_inputs, train_mean, train_train_covar, train_labels, likelihood, non_batch_train
    ):
        self.num_train = num_train
        self.train_inputs = train_inputs
        self.train_train_covar = train_train_covar
        self.train_mean = train_mean
        self.train_labels = train_labels
        self.likelihood = likelihood
        self.non_batch_train = non_batch_train
        self._last_test_train_covar = None
        mvn = self.likelihood(MultivariateNormal(train_mean, train_train_covar), train_inputs)
        self.lik_train_train_covar = mvn.lazy_covariance_matrix

    def __deepcopy__(self, memo):
        # deepcopying prediciton strategies of a model evaluated on inputs that require gradients fails
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
        if settings.detach_test_caches.on():
            return train_train_covar_inv_root.detach()
        else:
            return train_train_covar_inv_root

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        """
        Computes :math:`K_{X^{*}X} S` given a precomputed cache
        Where :math:`S` is a tensor such that :math:`SS^{\\top} = (K_{XX} + \sigma^2 I)^{-1}`

        Args:
            precomputed_cache (:obj:`torch.tensor`): What was computed in _exact_predictive_covar_inv_quad_form_cache
            test_train_covar (:obj:`torch.tensor`): The observed noise (from the likelihood)

        Returns
            :obj:`~gpytorch.lazy.LazyTensor`: :math:`K_{X^{*}X} S`
        """
        # Here the precomputed cache represents S,
        # where S S^T = (K_XX + sigma^2 I)^-1
        return test_train_covar.matmul(precomputed_cache)

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output):
        """
        Returns a new PredictionStrategy that incorporates the specified inputs and targets as new training data.

        This method is primary responsible for updating the mean and covariance caches. To add fantasy data to a
        GP model, use the :meth:`~gpytorch.models.ExactGP.get_fantasy_model` method.

        Args:
            - :attr:`inputs` (Tensor `m x d` or `b x m x d`): Locations of fantasy observations.
            - :attr:`targets` (Tensor `m` or `b x m`): Labels of fantasy observations.
            - :attr:`full_inputs` (Tensor `n+m x d` or `b x n+m x d`): Training data concatenated with fantasy inputs
            - :attr:`full_targets` (Tensor `n+m` or `b x n+m`): Training labels concatenated with fantasy labels.
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
        mvn = self.likelihood(MultivariateNormal(fant_mean, fant_fant_covar), inputs)
        fant_fant_covar = mvn.covariance_matrix

        fant_train_covar = delazify(full_covar[..., num_train:, :num_train])

        self.fantasy_inputs = inputs
        self.fantasy_targets = targets

        """
        Compute a new mean cache given the old mean cache.

        We have \\alpha = K^{-1}y, and we want to solve [K U; U' S][a; b] = [y; y_f], where U' is fant_train_covar,
        S is fant_fant_covar, and y_f is (targets - fant_mean)

        To do this, we solve the bordered linear system of equations for [a; b]:
            AQ = U  # Q = fant_solve
            [S - U'Q]b = y_f - U'\\alpha   ==> b = [S - U'Q]^{-1}(y_f - U'\\alpha)
            a = \\alpha - Qb
        """
        # Get cached K inverse decomp. (or compute if we somehow don't already have the covariance cache)
        K_inverse = self.lik_train_train_covar.root_inv_decomposition()
        fant_solve = K_inverse.matmul(fant_train_covar.transpose(-2, -1))

        # Solve for "b", the lower portion of the *new* \\alpha corresponding to the fantasy points.
        schur_complement = fant_fant_covar - fant_train_covar.matmul(fant_solve)
        small_system_rhs = targets - fant_mean - fant_train_covar.matmul(self.mean_cache)
        # Schur complement of a spd matrix is guaranteed to be positive definite
        if small_system_rhs.requires_grad or schur_complement.requires_grad:
            # TODO: Delete this part of the if statement when PyTorch implements cholesky_solve derivative.
            fant_cache_lower = torch.gesv(small_system_rhs.unsqueeze(-1), schur_complement)[0]
        else:
            fant_cache_lower = cholesky_solve(small_system_rhs, torch.cholesky(schur_complement))

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
        L = delazify(self.lik_train_train_covar.root_decomposition().root)
        m, n = L.shape[-2:]

        lower_left = fant_train_covar.matmul(L_inverse)
        schur_root = torch.cholesky(fant_fant_covar - lower_left.matmul(lower_left.transpose(-2, -1)))
        upper_right = torch.zeros(m, schur_root.size(-1), device=L.device, dtype=L.dtype)

        # Form new root Z = [L 0; lower_left schur_root]
        num_fant = schur_root.size(-2)
        m, n = L.shape[-2:]
        new_root = torch.zeros(*batch_shape, m + num_fant, n + num_fant, device=L.device, dtype=L.dtype)
        new_root[..., :m, :n] = L
        new_root[..., :m, n:] = upper_right
        new_root[..., m:, :n] = lower_left
        new_root[..., m:, n:] = schur_root

        # Use pseudo-inverse of Z as new inv root
        # TODO: Replace pseudo-inverse calculation with something more stable than normal equations once
        # one of torch.svd, torch.qr, or torch.pinverse works in batch mode.
        cap_mat = new_root.transpose(-2, -1).matmul(new_root)
        if cap_mat.requires_grad or new_root.requires_grad:
            # TODO: Delete this part of the if statement when PyTorch implements cholesky_solve derivative.
            new_covar_cache = torch.gesv(new_root.transpose(-2, -1), cap_mat)[0].transpose(-2, -1)
        else:
            new_covar_cache = cholesky_solve(new_root.transpose(-2, -1), torch.cholesky(cap_mat))
            new_covar_cache = new_covar_cache.transpose(-2, -1)

        # Create new DefaultPredictionStrategy object
        new_num_train = full_inputs[0].size(len(batch_shape))
        fant_strat = self.__class__(
            num_train=new_num_train,
            train_inputs=full_inputs,
            train_mean=full_mean,
            train_train_covar=full_covar,
            train_labels=full_targets,
            likelihood=self.likelihood,
            non_batch_train=(len(batch_shape) == 0),
        )
        setattr(fant_strat, "_memoize_cache", {"mean_cache": fant_mean_cache, "covar_cache": new_covar_cache})

        return fant_strat

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        train_mean = self.train_mean
        train_labels = self.train_labels
        train_inputs = self.train_inputs

        if self.non_batch_train and self.train_train_covar.dim() == 3:
            train_train_covar = self.train_train_covar[0]
        else:
            train_train_covar = self.train_train_covar

        if self.non_batch_train and train_mean.dim() == 2:
            train_mean = train_mean[0]
            train_labels = train_labels[0]
            train_inputs = tuple(ti[0] for ti in train_inputs)

        mvn = self.likelihood(MultivariateNormal(train_mean, train_train_covar), train_inputs)

        train_mean, train_train_covar = mvn.mean, mvn.lazy_covariance_matrix

        train_labels_offset = train_labels - train_mean

        if self.train_train_covar.dim() == 3:
            # Batch mode
            train_labels_offset = train_labels_offset.unsqueeze(-1)
            mean_cache = train_train_covar.inv_matmul(train_labels_offset).squeeze(-1)
        else:
            # Standard mode
            mean_cache = train_train_covar.inv_matmul(train_labels_offset)

        if settings.detach_test_caches.on():
            return mean_cache.detach()
        else:
            return mean_cache

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        train_train_covar = self.lik_train_train_covar

        if self.non_batch_train and train_train_covar.dim() == 3:
            train_train_covar_inv_root = delazify(train_train_covar[0].root_inv_decomposition().root)
        else:
            train_train_covar_inv_root = delazify(train_train_covar.root_inv_decomposition().root)

        return self._exact_predictive_covar_inv_quad_form_cache(train_train_covar_inv_root, self._last_test_train_covar)

    def exact_predictive_mean(self, test_mean, test_train_covar):
        """
        Computes the posterior predictive covariance of a GP

        Args:
            test_mean (:obj:`torch.tensor`): The test prior mean
            test_train_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test and train inputs

        Returns:
            :obj:`torch.tensor`: The predictive posterior mean of the test points
        """
        precomputed_cache = self.mean_cache

        if self.train_train_covar.dim() == 3:
            res = test_train_covar.matmul(
                precomputed_cache.expand(*test_train_covar.shape[:-2], test_train_covar.shape[-1]).unsqueeze(-1)
            ).squeeze(-1)
        else:
            if self.non_batch_train and precomputed_cache.dim() == 2:
                precomputed_cache = precomputed_cache[0]
            res = test_train_covar.matmul(precomputed_cache)

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
            if settings.detach_test_caches.on():
                train_train_covar = self.likelihood(
                    MultivariateNormal(torch.zeros(1), self.train_train_covar), self.train_inputs
                ).lazy_covariance_matrix.detach()
            else:
                train_train_covar = self.likelihood(
                    MultivariateNormal(torch.zeros(1), self.train_train_covar), self.train_inputs
                ).lazy_covariance_matrix

            test_train_covar = delazify(test_train_covar)
            train_test_covar = test_train_covar.transpose(-1, -2)
            covar_correction_rhs = train_train_covar.inv_matmul(train_test_covar).mul(-1)
            if torch.is_tensor(test_test_covar):
                return lazify(test_test_covar + test_train_covar @ covar_correction_rhs)
            else:
                return test_test_covar + MatmulLazyTensor(test_train_covar, covar_correction_rhs)

        precomputed_cache = self.covar_cache
        covar_inv_quad_form_root = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache,
                                                                                   test_train_covar)
        if torch.is_tensor(test_test_covar):
            return lazify(
                torch.add(test_test_covar, -1, covar_inv_quad_form_root @ covar_inv_quad_form_root.transpose(-1, -2))
            )
        else:
            return test_test_covar + MatmulLazyTensor(
                covar_inv_quad_form_root, covar_inv_quad_form_root.transpose(-1, -2).mul(-1)
            )


@register_prediction_strategy(InterpolatedLazyTensor)
class InterpolatedPredictionStrategy(DefaultPredictionStrategy):
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

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output):
        raise NotImplementedError(
            "Fantasy observation updates not yet supported for models using InterpolatedLazyTensors"
        )

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        train_interp_indices = self.train_train_covar.left_interp_indices
        train_interp_values = self.train_train_covar.left_interp_values

        mvn = self.likelihood(MultivariateNormal(self.train_mean, self.train_train_covar), self.train_inputs)
        train_mean, train_train_covar = mvn.mean, mvn.lazy_covariance_matrix

        train_train_covar_inv_labels = train_train_covar.inv_matmul((self.train_labels - train_mean).unsqueeze(-1))

        # New root factor
        base_size = self.train_train_covar.base_lazy_tensor.size(-1)
        mean_cache = self.train_train_covar.base_lazy_tensor.matmul(
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
        grv = MultivariateNormal(torch.zeros(1), self.train_train_covar)
        train_train_covar = self.likelihood(grv, self.train_inputs).lazy_covariance_matrix

        train_interp_indices = self.train_train_covar.left_interp_indices
        train_interp_values = self.train_train_covar.left_interp_values

        # Get probe vectors for inverse root
        num_probe_vectors = settings.fast_pred_var.num_probe_vectors()
        batch_size = train_interp_indices.size(0)
        n_inducing = self.train_train_covar.base_lazy_tensor.size(-1)
        vector_indices = torch.randperm(n_inducing).type_as(train_interp_indices)
        probe_vector_indices = vector_indices[:num_probe_vectors]
        test_vector_indices = vector_indices[num_probe_vectors : 2 * num_probe_vectors]

        probe_interp_indices = probe_vector_indices.unsqueeze(1)
        probe_test_interp_indices = test_vector_indices.unsqueeze(1)
        dtype = self.train_train_covar.dtype
        device = self.train_train_covar.device
        probe_interp_values = torch.ones(num_probe_vectors, 1, dtype=dtype, device=device)
        if train_interp_indices.ndimension() == 3:
            probe_interp_indices = probe_interp_indices.unsqueeze(0).expand(batch_size, num_probe_vectors, 1)
            probe_test_interp_indices = probe_test_interp_indices.unsqueeze(0)
            probe_test_interp_indices = probe_test_interp_indices.expand(batch_size, num_probe_vectors, 1)
            probe_interp_values = probe_interp_values.unsqueeze(0).expand(batch_size, num_probe_vectors, 1)

        probe_vectors = InterpolatedLazyTensor(
            self.train_train_covar.base_lazy_tensor,
            train_interp_indices,
            train_interp_values,
            probe_interp_indices,
            probe_interp_values,
        ).evaluate()
        test_vectors = InterpolatedLazyTensor(
            self.train_train_covar.base_lazy_tensor,
            train_interp_indices,
            train_interp_values,
            probe_test_interp_indices,
            probe_interp_values,
        ).evaluate()

        # Get inverse root
        train_train_covar_inv_root = train_train_covar.root_inv_decomposition(probe_vectors, test_vectors).root
        train_train_covar_inv_root = train_train_covar_inv_root.evaluate()

        # New root factor
        root = self._exact_predictive_covar_inv_quad_form_cache(train_train_covar_inv_root, self._last_test_train_covar)

        # Precomputed factor
        if settings.fast_pred_samples.on():
            inside = self.train_train_covar.base_lazy_tensor + RootLazyTensor(root).mul(-1)
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
        if (settings.fast_pred_samples.on() and precomputed_cache[0] is None) or (
            settings.fast_pred_samples.off() and precomputed_cache[1] is None
        ):
            self._memoize_cache.pop("covar_cache")
            precomputed_cache = self.covar_cache

        # Compute the exact predictive posterior
        if settings.fast_pred_samples.on():
            res = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache[0], test_train_covar)
            res = RootLazyTensor(res)
        else:
            root = left_interp(test_interp_indices, test_interp_values, precomputed_cache[1])
            res = test_test_covar + RootLazyTensor(root).mul(-1)
        return res


@register_prediction_strategy(SumLazyTensor)
class SumPredictionStrategy(DefaultPredictionStrategy):
    @property
    def _sub_strategies(self):
        sub_strategies = []
        for lazy_tensor in self.train_train_covar.lazy_tensors:
            pred_strat = prediction_strategy(
                self.num_train,
                self.train_inputs,
                self.train_mean,
                lazy_tensor,
                self.train_labels,
                self.likelihood,
                self.non_batch_train,
            )
            sub_strategies.append(pred_strat)

        return sub_strategies

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
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
        test_train_covar = test_train_covar.evaluate_kernel()
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
