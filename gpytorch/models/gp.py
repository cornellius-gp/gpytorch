#!/usr/bin/env python3

from __future__ import annotations

from ..module import Module


class GP(Module):
    def __init__(self):
        super().__init__()
        # Bound-method load_state_dict pre-hooks registered by gpytorch.Module create a
        # reference cycle (module -> hooks -> bound method -> module) that delays GC of
        # large attributes such as ExactGP.train_inputs and can OOM on CUDA when models
        # are constructed repeatedly. Clearing after init breaks the cycle. See #2649.
        self._load_state_dict_pre_hooks.clear()
