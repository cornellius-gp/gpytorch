#!/usr/bin/env python3

import functools
from abc import ABC, abstractproperty
from copy import deepcopy

import torch

from .. import settings
from ..distributions import Delta, MultivariateNormal
from ..likelihoods import GaussianLikelihood
from ..models import ExactGP
from ..module import Module
from ..utils.memoize import add_to_cache, cached, clear_cache_hook


class _BaseExactGP(ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, mean_module, covar_module):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


def _add_cache_hook(tsr, pred_strat):
    if tsr.grad_fn is not None:
        wrapper = functools.partial(clear_cache_hook, pred_strat)
        functools.update_wrapper(wrapper, clear_cache_hook)
        tsr.grad_fn.register_hook(wrapper)
    return tsr


class _VariationalStrategy(Module, ABC):
    """
    Abstract base class for all Variational Strategies.
    """

    has_fantasy_strategy = False

    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=True):
        super().__init__()

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        if learn_inducing_locations:
            self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("inducing_points", inducing_points)

        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    def _clear_cache(self):
        clear_cache_hook(self)

    def _expand_inputs(self, x, inducing_points):
        """
        Pre-processing step in __call__ to make x the same batch_shape as the inducing points
        """
        batch_shape = torch.broadcast_shapes(inducing_points.shape[:-2], x.shape[:-2])
        inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
        x = x.expand(*batch_shape, *x.shape[-2:])
        return x, inducing_points

    @abstractproperty
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`p( \mathbf u)`
        """
        raise NotImplementedError

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        return self._variational_distribution()

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        :param torch.Tensor x: Locations :math:`\mathbf X` to get the
            variational posterior of the function values at.
        :param torch.Tensor inducing_points: Locations :math:`\mathbf Z` of the inducing points
        :param torch.Tensor inducing_values: Samples of the inducing function values :math:`\mathbf u`
            (or the mean of the distribution :math:`q(\mathbf u)` if q is a Gaussian.
        :param ~linear_operator.operators.LinearOperator variational_inducing_covar: If
            the distribuiton :math:`q(\mathbf u)` is
            Gaussian, then this variable is the covariance matrix of that Gaussian.
            Otherwise, it will be None.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`q( \mathbf f(\mathbf X))`
        """
        raise NotImplementedError

    def kl_divergence(self):
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.

        :rtype: torch.Tensor
        """
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence

    @cached(name="amortized_exact_gp")
    def amortized_exact_gp(self, mean_module=None, covar_module=None):
        mean_module = self.model.mean_module if mean_module is None else mean_module
        covar_module = self.model.covar_module if covar_module is None else covar_module

        with torch.no_grad():
            # from here on down, we refer to the inducing points as pseudo_inputs
            pseudo_target_covar, pseudo_target_mean = self.pseudo_points
            pseudo_inputs = self.inducing_points.detach()
            if pseudo_inputs.ndim < pseudo_target_mean.ndim:
                pseudo_inputs = pseudo_inputs.expand(*pseudo_target_mean.shape[:-2], *pseudo_inputs.shape)
            # TODO: add flag for conditioning into SGPR after building fantasy strategy for SGPR
            new_covar_module = deepcopy(covar_module)

            # update inducing mean if necessary
            pseudo_target_mean = pseudo_target_mean.squeeze() + mean_module(pseudo_inputs)

            inducing_exact_model = _BaseExactGP(
                pseudo_inputs,
                pseudo_target_mean,
                mean_module=deepcopy(mean_module),
                covar_module=new_covar_module,
                likelihood=deepcopy(self.model.likelihood),
            )

            # now fantasize around this model
            # as this model is new, we need to compute a posterior to construct the prediction strategy
            # which uses the likelihood pseudo caches
            faked_points = torch.randn(
                *pseudo_target_mean.shape[:-2],
                1,
                pseudo_inputs.shape[-1],
                device=pseudo_inputs.device,
                dtype=pseudo_inputs.dtype,
            )
            inducing_exact_model.eval()
            _ = inducing_exact_model(faked_points)

            # then we overwrite the likelihood to take into account the multivariate normal term
            pred_strat = inducing_exact_model.prediction_strategy
            pred_strat._memoize_cache = {}
            with torch.no_grad():
                updated_lik_train_train_covar = pred_strat.train_prior_dist.lazy_covariance_matrix + pseudo_target_covar
                pred_strat.lik_train_train_covar = updated_lik_train_train_covar

            # do the mean cache because the mean cache doesn't solve against lik_train_train_covar
            train_mean = inducing_exact_model.mean_module(*inducing_exact_model.train_inputs)
            train_labels_offset = (inducing_exact_model.prediction_strategy.train_labels - train_mean).unsqueeze(-1)
            mean_cache = updated_lik_train_train_covar.solve(train_labels_offset).squeeze(-1)
            mean_cache = _add_cache_hook(mean_cache, inducing_exact_model.prediction_strategy)
            add_to_cache(pred_strat, "mean_cache", mean_cache)
            # TODO: check to see if we need to do the covar_cache?

            inducing_exact_model.prediction_strategy = pred_strat
        return inducing_exact_model

    def pseudo_points(self):
        raise NotImplementedError("Each variational strategy must implement its own pseudo points method")

    def get_fantasy_model(
        self,
        inputs,
        targets,
        mean_module=None,
        covar_module=None,
        **kwargs,
    ):
        r"""
        Performs the online variational conditioning (OVC) strategy of Maddox et al, '21 to return
        an exact GP model that incorporates the inputs and targets alongside the variational model's inducing
        points and targets.

        Currently, instead of directly updating the variational parameters (and inducing points), we instead
        return an ExactGP model rather than an updated variational GP model. This is done primarily for
        numerical stability.

        Unlike the ExactGP's call for get_fantasy_model, we enable options for mean_module and covar_module
        that allow specification of the mean / covariance. We expect that either the mean and covariance
        modules are attributes of the model itself called mean_module and covar_module respectively OR that you
        pass them into this method explicitly.

        :param torch.Tensor inputs: (`b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`) Locations of fantasy
            observations.
        :param torch.Tensor targets: (`b1 x ... x bk x m` or `f x b1 x ... x bk x m`) Labels of fantasy observations.
        :param torch.nn.Module mean_module: torch module describing the mean function of the GP model. Optional if
            `mean_module` is already an attribute of the variational GP.
        :param torch.nn.Module covar_module: torch module describing the covariance function of the GP model. Optional
            if `covar_module` is already an attribute of the variational GP.
        :return: An `ExactGP` model with `k + m` training examples, where the `m` fantasy examples have been added
            and all test-time caches have been updated. We assume that there are `k` inducing points in this variational
            GP. Note that we return an `ExactGP` rather than a variational GP.
        :rtype: ~gpytorch.models.ExactGP

        Reference: "Conditioning Sparse Variational Gaussian Processes for Online Decision-Making,"
            Maddox, Stanton, Wilson, NeurIPS, '21
            https://papers.nips.cc/paper/2021/hash/325eaeac5bef34937cfdc1bd73034d17-Abstract.html
        """

        # currently, we only support fantasization for CholeskyVariationalDistribution and
        # whitened / unwhitened variational strategies
        if not self.has_fantasy_strategy:
            raise NotImplementedError(
                "No fantasy model support for ",
                self.__name__,
                ". Only VariationalStrategy and UnwhitenedVariationalStrategy are currently supported.",
            )
        if not isinstance(self.model.likelihood, GaussianLikelihood):
            raise NotImplementedError(
                "No fantasy model support for ",
                self.model.likelihood,
                ". Only GaussianLikelihoods are currently supported.",
            )
        # we assume that either the user has given the model a mean_module and a covar_module
        # or that it will be passed into the get_fantasy_model function. we check for these.
        if mean_module is None:
            mean_module = getattr(self.model, "mean_module", None)
            if mean_module is None:
                raise ModuleNotFoundError(
                    "Either you must provide a mean_module as input to get_fantasy_model",
                    "or it must be an attribute of the model called mean_module.",
                )
        if covar_module is None:
            covar_module = getattr(self.model, "covar_module", None)
            if covar_module is None:
                # raise an error
                raise ModuleNotFoundError(
                    "Either you must provide a covar_module as input to get_fantasy_model",
                    "or it must be an attribute of the model called covar_module.",
                )

        # first we construct an exact model over the inducing points with the inducing covariance
        # matrix
        inducing_exact_model = self.amortized_exact_gp(mean_module=mean_module, covar_module=covar_module)

        # then we update this model by adding in the inputs and pseudo targets
        # finally we fantasize wrt targets
        fantasy_model = inducing_exact_model.get_fantasy_model(inputs, targets, **kwargs)
        fant_pred_strat = fantasy_model.prediction_strategy

        # first we update the lik_train_train_covar
        # do the mean cache again because the mean cache resets the likelihood forward
        train_mean = fantasy_model.mean_module(*fantasy_model.train_inputs)
        train_labels_offset = (fant_pred_strat.train_labels - train_mean).unsqueeze(-1)
        fantasy_lik_train_root_inv = fant_pred_strat.lik_train_train_covar.root_inv_decomposition()
        mean_cache = fantasy_lik_train_root_inv.matmul(train_labels_offset).squeeze(-1)
        mean_cache = _add_cache_hook(mean_cache, fant_pred_strat)
        add_to_cache(fant_pred_strat, "mean_cache", mean_cache)
        # TODO: should we update the covar_cache?

        fantasy_model.prediction_strategy = fant_pred_strat
        return fantasy_model

    def __call__(self, x, prior=False, **kwargs):
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x, **kwargs)

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()
        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            self._variational_distribution.initialize_variational_distribution(prior_dist)
            self.variational_params_initialized.fill_(1)

        # Ensure inducing_points and x are the same size
        inducing_points = self.inducing_points
        if inducing_points.shape[:-2] != x.shape[:-2]:
            x, inducing_points = self._expand_inputs(x, inducing_points)

        # Get p(u)/q(u)
        variational_dist_u = self.variational_distribution

        # Get q(f)
        if isinstance(variational_dist_u, MultivariateNormal):
            return super().__call__(
                x,
                inducing_points,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                **kwargs,
            )
        elif isinstance(variational_dist_u, Delta):
            return super().__call__(
                x, inducing_points, inducing_values=variational_dist_u.mean, variational_inducing_covar=None, **kwargs
            )
        else:
            raise RuntimeError(
                f"Invalid variational distribuition ({type(variational_dist_u)}). "
                "Expected a multivariate normal or a delta distribution."
            )
