"""Abstract base class for Gaussian process approximation strategies."""

from __future__ import annotations

import abc
from collections import defaultdict

from typing import Iterable, Optional

from jaxtyping import Float
from torch import nn, Tensor

from ... import distributions, Module


class ApproximationStrategy(abc.ABC, Module):
    """Abstract base class for Gaussian process approximation strategies."""

    def __init__(self) -> None:
        super().__init__()

    def init_cache(
        self,
        model: Module,
        train_inputs: Optional[Float[Tensor, "N D"]] = None,
        train_targets: Optional[Float[Tensor, " N"]] = None,
    ) -> None:

        # Set model as an attribute of the ApproximationStrategy without registering it as a
        # submodule of ApproximationStrategy by bypassing Module.__setattr__ explicitly.
        object.__setattr__(self, "model", model)

        # Events triggering certain caches to clear
        self._clear_cache_triggers = defaultdict(list)

        self.train_inputs = train_inputs
        self.train_targets = train_targets

    def register_cached_quantity(
        self,
        name: str,
        tensor: Optional[Tensor] = None,
        persistent: bool = True,
        clear_cache_on: Optional[Iterable[str]] = ["backward", "train_inputs_set", "train_targets_set"],
        clear_cache_on_backward_of_params: Optional[Iterable[nn.Parameter]] = None,
    ) -> None:
        """Register a cached quantity used to save computation.

        Cached quantities are PyTorch buffers, meaning they are not considered trainable model parameters,
        but are part of the module's state. This behavior can be changed by setting ``persistent=False``.
        Caches can be cleared automatically based on certain function calls involving the model.

        Cached quantities can be accessed as attributes using their ``name``.

        :param name: Name of the cached quantity. The cached quantity can be accessed from this module using the
            given name.
        :param tensor: Cached quantity to be registered. If ``None``, then operations that run on buffers, such
            as cuda, are ignored. If ``None``, the cached quantity is not included in the module's ``state_dict``
            (unless it is set to a tensor later on).
        :param persistent: Whether the cached quantity is part of the module's ``state_dict``.
        :param clear_cache_on: When to automatically clear the cached quantity.
            Zero or more of: ``["backward", "set_train_inputs", "set_train_targets"]``.
        :param clear_cache_on_backward_of_params: If you need more control over which parameters of the model
            should trigger releasing the cache during a backward pass you can pass model parameters here.
            If you specify ``"backward"`` in ``clear_cache_on``, passing parameters here does nothing.
        """

        if self.model is None:
            raise AttributeError(
                "ApproximationStrategy is not associated with a model. "
                "You likely registered a cached quantity outside of or before calling "
                "`ApproximationStrategy.init_cache`."
            )

        # Register cached quantity as a PyTorch buffer
        self.register_buffer(
            name=name,
            tensor=(
                tensor.detach() if tensor is not None else None
            ),  # Ensure cached quantity is detached from the graph.
            persistent=persistent,
        )

        # Automatically clear cache on certain events
        if clear_cache_on is not None:
            for clear_cache_trigger in clear_cache_on:
                if clear_cache_trigger == "backward":
                    # Register backward hook to clear cache
                    for _, param in self.model.named_parameters():
                        param.register_hook(lambda _: self.__setattr__(name, None))
                else:
                    # Custom triggers to clear cache
                    self._clear_cache_triggers[clear_cache_trigger].append(name)

        # Clear cache only on backward for specific model parameters
        if clear_cache_on_backward_of_params is not None:
            for param in clear_cache_on_backward_of_params:
                param.register_hook(lambda _: self.__setattr__(name, None))

    def __setattr__(self, name: str, value: Tensor | nn.Module) -> None:

        # Ensure buffers / caches never require grad.
        if name in self._buffers.keys():
            if self._buffers[name] is not None:
                if self._buffers[name].requires_grad:
                    raise ValueError(
                        f"Trying to set buffer / cache `{name}`, which requires a gradient. "
                        "Make sure you .detach() cached quantities from the graph first."
                    )

        super().__setattr__(name, value)

    @property
    def train_inputs(self) -> Float[Tensor, "N D"]:
        return self._train_inputs

    @train_inputs.setter
    def train_inputs(self, value: Optional[Float[Tensor, "N D"]]):

        # Clear cached quantities which depend on the training inputs.
        for cached_quantity in self._clear_cache_triggers["set_train_inputs"]:
            self.__setattr__(cached_quantity, None)

        # Reshape train inputs into a 2D tensor in case a 1D tensor is passed.
        if value is None:
            self._train_inputs = value
        else:
            self._train_inputs = value.unsqueeze(-1) if value.ndimension() <= 1 else value

    @property
    def train_targets(self) -> Optional[Float[Tensor, " N"]]:
        return self._train_targets

    @train_targets.setter
    def train_targets(self, value: Optional[Float[Tensor, " N"]]):

        # Clear cached quantities which depend on the training targets.
        for cached_quantity in self._clear_cache_triggers["set_train_targets"]:
            self.__setattr__(cached_quantity, None)

        self._train_targets = value

    @abc.abstractmethod
    def posterior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:
        """Evaluate the approximate posterior distribution of the Gaussian process."""
        raise NotImplementedError
