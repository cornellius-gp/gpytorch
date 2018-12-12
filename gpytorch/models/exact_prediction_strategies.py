import torch
from .. import beta_features
from ..lazy import LazyTensor, InterpolatedLazyTensor, SumLazyTensor, MatmulLazyTensor, RootLazyTensor
from ..utils.interpolation import left_interp, left_t_interp

_PREDICTION_STRATEGY_REGISTRY = {}


def register_prediction_strategy(lazy_tsr_type):
    if not isinstance(lazy_tsr_type, type) and issubclass(lazy_tsr_type, LazyTensor):
        raise TypeError(f"register_prediction_strategy expects a LazyTensor subtype but got {lazy_tsr_type}")

    def decorator(cls):
        _PREDICTION_STRATEGY_REGISTRY[lazy_tsr_type] = cls
        return cls

    return decorator


def prediction_strategy(train_train_covar):
    cls = _PREDICTION_STRATEGY_REGISTRY.get(type(train_train_covar), DefaultPredictionStrategy)
    return cls(train_train_covar)


class DefaultPredictionStrategy(object):
    def __init__(self, train_train_covar):
        self.train_train_covar = train_train_covar

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
        return train_train_covar_inv_root.detach()

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

    def exact_predictive_mean(
        self,
        test_train_covar,
        full_mean,
        train_inputs,
        train_labels,
        num_train,
        likelihood,
        precomputed_cache=None,
        non_batch_train=False,
    ):
        """
        Computes the posterior predictive covariance of a GP

        Args:
            test_train_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test and train inputs
            full_mean (:obj:`torch.tensor`): the training and test prior means, stacked on top of each other
            train_inputs (:obj:`torch.tensor`): The training data inputs
            train_labels (:obj:`torch.tensor`): the training labels minus the training prior mean
            noise (:obj:`torch.tensor`): the observed noise (from the likelihood)
            precomputed_cache (optional): speeds up subsequent computations (default: None)

        Returns:
            :obj:`torch.tensor`: The predictive posterior mean of the test points
        """
        from ..distributions import MultivariateNormal

        if precomputed_cache is None:
            train_mean = full_mean.narrow(-1, 0, num_train)

            if non_batch_train and self.train_train_covar.dim() == 3:
                train_train_covar = self.train_train_covar[0]
            else:
                train_train_covar = self.train_train_covar

            train_mean = full_mean.narrow(-1, 0, train_train_covar.size(-1))
            if non_batch_train and train_mean.dim() == 2:
                train_mean = train_mean[0]
                train_labels = train_labels[0]
            mvn = likelihood(MultivariateNormal(train_mean, train_train_covar), train_inputs)

            train_mean, train_train_covar = mvn.mean, mvn.lazy_covariance_matrix

            train_labels_offset = train_labels - train_mean

            if self.train_train_covar.dim() == 3:
                # Batch mode
                train_labels_offset = train_labels_offset.unsqueeze(-1)
                precomputed_cache = train_train_covar.inv_matmul(train_labels_offset).squeeze(-1)
            else:
                # Standard mode
                precomputed_cache = train_train_covar.inv_matmul(train_labels_offset)

        test_mean = full_mean.narrow(-1, train_labels.size(-1), full_mean.size(-1) - train_labels.size(-1))

        if self.train_train_covar.dim() == 3:
            res = test_train_covar.matmul(precomputed_cache.unsqueeze(-1)).squeeze(-1)
        else:
            if non_batch_train and precomputed_cache.dim() == 2:
                precomputed_cache = precomputed_cache[0]
            res = test_train_covar.matmul(precomputed_cache)

        res = res + test_mean

        return res, precomputed_cache.detach()

    def exact_predictive_covar(
        self,
        test_train_covar,
        test_test_covar,
        train_inputs,
        num_train,
        likelihood,
        precomputed_cache=None,
        non_batch_train=False,
    ):
        """
        Computes the posterior predictive covariance of a GP

        Args:
            test_train_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test and train inputs
            test_test_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test inputs
            train_inputs (:obj:`torch.tensor`): The training data inputs
            num_train (int): The number of training points in the full covariance matrix
            noise (scalar): The observed noise (from the likelihood)
            precomputed_cache (optional): speeds up subsequent computations (default: None)
            non_batch_train (bool, optional): True if the training data was not batch mode.

        Returns:
            :obj:`gpytorch.lazy.LazyTensor`: A LazyTensor representing the predictive posterior covariance of the
                                               test points
        """
        from ..distributions import MultivariateNormal

        train_train_covar = likelihood(
            MultivariateNormal(torch.zeros(1), self.train_train_covar), train_inputs
        ).lazy_covariance_matrix
        if not beta_features.fast_pred_var.on():
            test_train_covar = test_train_covar.evaluate()
            train_test_covar = test_train_covar.transpose(-1, -2)
            covar_correction_rhs = train_train_covar.inv_matmul(train_test_covar).mul(-1)
            res = test_test_covar + MatmulLazyTensor(test_train_covar, covar_correction_rhs)
            return res, None

        if precomputed_cache is None:
            if non_batch_train and train_train_covar.dim() == 3:
                train_train_covar_inv_root = train_train_covar[0].root_inv_decomposition().root.evaluate()
            else:
                train_train_covar_inv_root = train_train_covar.root_inv_decomposition().root.evaluate()
            precomputed_cache = self._exact_predictive_covar_inv_quad_form_cache(
                train_train_covar_inv_root, test_train_covar
            )

        covar_inv_quad_form_root = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache, test_train_covar)
        res = test_test_covar + RootLazyTensor(covar_inv_quad_form_root).mul(-1)
        return res, precomputed_cache


@register_prediction_strategy(InterpolatedLazyTensor)
class InterpolatedPredictionStrategy(DefaultPredictionStrategy):
    def __init__(self, train_train_covar):
        if not isinstance(train_train_covar, InterpolatedLazyTensor):
            raise TypeError(
                "InterpolatedPredictionStrategy can only be used with InterpolatedLazyTensors, "
                f"but got {type(train_train_covar)}"
            )

        super(InterpolatedPredictionStrategy, self).__init__(train_train_covar)

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

    def exact_predictive_mean(
        self,
        test_train_covar,
        full_mean,
        train_inputs,
        train_labels,
        num_train,
        likelihood,
        precomputed_cache=None,
        non_batch_train=False,
    ):
        from ..distributions import MultivariateNormal

        if precomputed_cache is None:
            train_interp_indices = self.train_train_covar.left_interp_indices
            train_interp_values = self.train_train_covar.left_interp_values

            train_mean = full_mean.narrow(-1, 0, train_labels.size(-1))

            mvn = likelihood(MultivariateNormal(train_mean, self.train_train_covar), train_inputs)
            train_mean, train_train_covar = mvn.mean, mvn.lazy_covariance_matrix

            train_train_covar_inv_labels = train_train_covar.inv_matmul((train_labels - train_mean).unsqueeze(-1))

            # New root factor
            base_size = self.train_train_covar.base_lazy_tensor.size(-1)
            precomputed_cache = self.train_train_covar.base_lazy_tensor.matmul(
                left_t_interp(train_interp_indices, train_interp_values, train_train_covar_inv_labels, base_size)
            )

            # Prevent backprop through this variable
            precomputed_cache = precomputed_cache.detach()

        # Compute the exact predictive posterior
        n_test = test_train_covar.size(-2)

        test_train_covar = test_train_covar.evaluate_kernel()

        test_mean = full_mean.narrow(-1, num_train, n_test)
        test_interp_indices = test_train_covar.left_interp_indices
        test_interp_values = test_train_covar.left_interp_values
        res = left_interp(test_interp_indices, test_interp_values, precomputed_cache).squeeze(-1) + test_mean
        return res, precomputed_cache

    def exact_predictive_covar(
        self,
        test_train_covar,
        test_test_covar,
        train_inputs,
        num_train,
        likelihood,
        precomputed_cache=None,
        non_batch_train=False,
    ):
        from ..distributions import MultivariateNormal

        if not beta_features.fast_pred_var.on() and not beta_features.fast_pred_samples.on():
            return super(InterpolatedPredictionStrategy, self).exact_predictive_covar(
                test_train_covar, test_test_covar, train_inputs, num_train, likelihood, precomputed_cache
            )

        train_interp_indices = self.train_train_covar.left_interp_indices
        train_interp_values = self.train_train_covar.left_interp_values

        test_train_covar = test_train_covar.evaluate_kernel()
        test_test_covar = test_test_covar.evaluate_kernel()

        test_interp_indices = test_train_covar.left_interp_indices
        test_interp_values = test_train_covar.left_interp_values

        if (
            precomputed_cache is None
            or (beta_features.fast_pred_samples.on() and precomputed_cache[0] is None)
            or (not beta_features.fast_pred_samples.on() and precomputed_cache[1] is None)
        ):
            # Get inverse root

            grv = MultivariateNormal(torch.zeros(1), self.train_train_covar)
            train_train_covar = likelihood(grv, train_inputs).lazy_covariance_matrix

            # Get probe vectors for inverse root
            num_probe_vectors = beta_features.fast_pred_var.num_probe_vectors()
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
            root = self._exact_predictive_covar_inv_quad_form_cache(train_train_covar_inv_root, test_train_covar)

            # Precomputed factor
            if beta_features.fast_pred_samples.on():
                inside = self.train_train_covar.base_lazy_tensor + RootLazyTensor(root).mul(-1)
                inside_root = inside.root_decomposition().root.evaluate()
                # Prevent backprop through this variable
                inside_root = inside_root.detach()
                precomputed_cache = inside_root, None
            else:
                # Prevent backprop through this variable
                root = root.detach()
                precomputed_cache = None, root

        # Compute the exact predictive posterior
        if beta_features.fast_pred_samples.on():
            res = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache[0], test_train_covar)
            res = RootLazyTensor(res)
        else:
            root = left_interp(test_interp_indices, test_interp_values, precomputed_cache[1])
            res = test_test_covar + RootLazyTensor(root).mul(-1)
        return res, precomputed_cache


@register_prediction_strategy(SumLazyTensor)
class SumPredictionStrategy(DefaultPredictionStrategy):
    def __init__(self, train_train_covar):
        if not isinstance(train_train_covar, SumLazyTensor):
            raise TypeError(
                f"SumPredictionStrategy can only be used with SumLazyTensors, but got {type(train_train_covar)}"
            )

        super(SumPredictionStrategy, self).__init__(train_train_covar)

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        test_train_covar = test_train_covar.evaluate_kernel()
        if not isinstance(test_train_covar, SumLazyTensor):
            return super(SumPredictionStrategy, self)._exact_predictive_covar_inv_quad_form_cache(
                train_train_covar_inv_root, test_train_covar
            )
        else:
            return tuple(
                prediction_strategy(lazy_tensor)._exact_predictive_covar_inv_quad_form_cache(
                    train_train_covar_inv_root, test_train_covar_comp
                )
                for lazy_tensor, test_train_covar_comp in zip(
                    self.train_train_covar.lazy_tensors, test_train_covar.lazy_tensors
                )
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
                prediction_strategy(lazy_tensor)._exact_predictive_covar_inv_quad_form_root(
                    cache_comp, test_train_covar_comp
                )
                for lazy_tensor, cache_comp, test_train_covar_comp in zip(
                    self.train_train_covar.lazy_tensors,
                    precomputed_cache,
                    test_train_covar.evaluate_kernel().lazy_tensors,
                )
            )
