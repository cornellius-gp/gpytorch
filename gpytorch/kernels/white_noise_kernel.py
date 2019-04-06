#!/usr/bin/env python3

from . import Kernel


class WhiteNoiseKernel(Kernel):
    """
    The WhiteNoiseKernel has been hard deprecated due to incorrect behavior in certain cases.
    For equivalent functionality, please use a FixedNoiseGaussianLikelihood.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError("WhiteNoiseKernel is now hard deprecated. Use a FixedNoiseLikelihood instead.")
