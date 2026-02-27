#!/usr/bin/env python3

from __future__ import annotations

try:
    from ._pyro_mixin import _PyroMixin
    from .pyro_gp import PyroGP
except ImportError:

    class PyroGP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a PyroGP because you dont have Pyro installed.")

    class _PyroMixin:
        def pyro_factors(self, *args, **kwargs):
            raise RuntimeError("Cannot call `pyro_factors` because you dont have Pyro installed.")

        def pyro_guide(self, *args, **kwargs):
            raise RuntimeError("Cannot call `pyro_sample` because you dont have Pyro installed.")

        def pyro_model(self, *args, **kwargs):
            raise RuntimeError("Cannot call `pyro_sample` because you dont have Pyro installed.")


__all__ = ["PyroGP", "_PyroMixin"]
