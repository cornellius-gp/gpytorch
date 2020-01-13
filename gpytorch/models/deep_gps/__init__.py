#!/usr/bin/env python3

import warnings

from .deep_gp import DeepGP, DeepGPLayer, DeepLikelihood


# Deprecated for 1.0 release
class AbstractDeepGP(DeepGP):
    def __init__(self, *args, **kwargs):
        warnings.warn("AbstractDeepGP has been renamed to DeepGP.", DeprecationWarning)
        super().__init__(*args, **kwargs)


# Deprecated for 1.0 release
class AbstractDeepGPLayer(DeepGPLayer):
    def __init__(self, *args, **kwargs):
        warnings.warn("AbstractDeepGPLayer has been renamed to DeepGPLayer.", DeprecationWarning)
        super().__init__(*args, **kwargs)


__all__ = ["DeepGPLayer", "DeepGP", "AbstractDeepGPLayer", "AbstractDeepGP", "DeepLikelihood"]
