#!/usr/bin/env python3

from __future__ import annotations

from .bayesian_gplvm import BayesianGPLVM
from .latent_variable import MAPLatentVariable, PointLatentVariable, VariationalLatentVariable

__all__ = ["BayesianGPLVM", "PointLatentVariable", "MAPLatentVariable", "VariationalLatentVariable"]
