"""Abstract base class for Gaussian process approximation strategies."""

from __future__ import annotations

import abc
from collections import defaultdict

from typing import Iterable, Literal, Optional, Union

from jaxtyping import Float
from linear_operator import operators
from torch import nn, Tensor

from ... import distributions, Module, settings


class ApproximationStrategy(abc.ABC, Module):
    """Abstract base class for Gaussian process approximation strategies."""

    def __init__(self) -> None:
        super().__init__()

        # Flag determining whether the cache has been initialized
        self._cache_initialized = False

        # Events triggering certain caches to clear
        self._clear_cache_triggers = defaultdict(list)

    def init_cache(
        self,
        model: Module,
    ) -> None:

        # Set model as an attribute of the ApproximationStrategy without registering it as a
        # submodule of ApproximationStrategy by bypassing Module.__setattr__ explicitly.
        object.__setattr__(self, "model", model)

        self._cache_initialized = True

    def register_cached_quantity(
        self,
        name: str,
        quantity: Optional[Union[Tensor, operators.LinearOperator]] = None,
        persistent: bool = True,
        clear_cache_on: Optional[Iterable[Literal["backward", "set_train_inputs", "set_train_targets"]]] = [
            "backward",
            "set_train_inputs",
            "set_train_targets",
            # TODO: if needed, add "train_mode", "eval_mode"
        ],
        clear_cache_on_backward_of_params: Optional[Iterable[nn.Parameter]] = None,
    ) -> None:
        """Register a cached quantity used to save computation.

        Cached quantities are PyTorch buffers, meaning they are not considered trainable model parameters,
        but are part of the module's state. This behavior can be changed by setting ``persistent=False``.
        Caches can be cleared automatically based on certain function calls involving the model.

        Cached quantities can be accessed as attributes using their ``name``.

        :param name: Name of the cached quantity. The cached quantity can be accessed from this module using the
            given name.
        :param quantity: Cached quantity to be registered. If ``None``, then operations that run on buffers, such
            as cuda, are ignored. If ``None``, the cached quantity is not included in the module's ``state_dict``
            (unless it is set to a quantity later on).
        :param persistent: Whether the cached quantity is part of the module's ``state_dict``.
        :param clear_cache_on: What events / function calls trigger clearing the cached quantity.
        :param clear_cache_on_backward_of_params: If you need more control over which parameters of the model
            should trigger releasing the cache during a backward pass you can pass model parameters here.
            If you specify ``"backward"`` in ``clear_cache_on``, passing parameters here does nothing.
        """

        if not self._cache_initialized:
            raise AttributeError(
                "Cannot register a cached quantity without initializing the cache via "
                "`ApproximationStrategy.init_cache(model)`."
                "Make sure you register cached quantities in `MyApproximationStrategy.init_cache` "
                "(or after calling `MyApproximationStrategy.init_cache`)."
            )

        # Register cached quantity as a PyTorch buffer
        self.register_buffer(
            name=name,
            # Ensure cached quantity is detached from the graph, but requires gradient if the original quantity does.
            # See also the Buffer class in https://github.com/pytorch/pytorch/pull/125971
            tensor=quantity.detach().requires_grad_(quantity.requires_grad) if quantity is not None else None,
            persistent=persistent,
        )

        # Automatically clear cache on certain events
        if clear_cache_on is not None:
            for clear_cache_trigger in clear_cache_on:
                if clear_cache_trigger == "backward":
                    # Register backward hook to clear cache
                    for param_name, param in self.model.named_parameters():
                        if param.requires_grad:

                            def clear_cache(_):
                                if settings.verbose_caches.on() and self.__getattr__(name) is not None:
                                    settings.verbose_caches.logger.debug(
                                        f"Clearing cache of ApproximationStrategy: '{self.__class__.__name__}.{name}' "
                                        f"via backward hook registered to {param_name}."
                                    )
                                self.__setattr__(name, None)

                            param.register_hook(clear_cache)
                else:
                    # Custom triggers to clear cache
                    self._clear_cache_triggers[clear_cache_trigger].append(name)

        # Clear cache only on backward for specific model parameters
        if clear_cache_on_backward_of_params is not None:
            for param in clear_cache_on_backward_of_params:
                if param.requires_grad:

                    def clear_cache(_):
                        if settings.verbose_caches.on() and self.__getattr__(name) is not None:
                            settings.verbose_caches.logger.debug(
                                f"Clearing cache of ApproximationStrategy: '{self.__class__.__name__}.{name}' "
                                "via backward hook registered to a model parameter."
                            )
                        self.__setattr__(name, None)

                    param.register_hook(clear_cache)

    @abc.abstractmethod
    def posterior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the approximate posterior distribution of the Gaussian process."""
        raise NotImplementedError
