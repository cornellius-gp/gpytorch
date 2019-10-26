#!/usr/bin/env python3

try:
    from .pyro_gp import PyroGP
except ImportError:
    class PyroGP(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a PyroGP because you dont have Pyro installed.")


__all__ = [
    "PyroGP",
]
