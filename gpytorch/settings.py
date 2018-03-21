from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class _feature_flag(object):
    _state = False

    @classmethod
    def on(cls):
        return cls._state

    @classmethod
    def _set_state(cls, state):
        cls._state = state

    def __init__(self, state=True):
        self.prev = self.__class__.on()
        self.state = state

    def __enter__(self):
        self.__class__._set_state(self.state)

    def __exit__(self, *args):
        self.__class__._set_state(self.prev)
        return False


class _value_context(object):
    _global_value = None

    @classmethod
    def value(cls):
        return cls._global_value

    @classmethod
    def _set_value(cls, value):
        cls._global_value = value

    def __init__(self, value):
        self._orig_value = self.__class__.value()
        self._instance_value = value

    def __enter__(self, ):
        self.__class__._set_value(self._instance_value)

    def __exit__(self, *args):
        self.__class__._set_value(self._orig_value)
        return False


class max_cg_iterations(_value_context):
    """
    The maximum number of conjugate gradient iterations to perform (when computing matrix solves)
    More values results in more accurate solves
    Default: 20
    """
    _global_value = 20


class max_lanczos_iterations(_value_context):
    """
    The maximum number of Lanczos iterations to perform
    This is used when 1) computing variance estiamtes 2) when drawing from MVNs, or
    3) for kernel multiplication
    More values results in higher accuracy
    Default: 100
    """
    _global_value = 100


class max_lanczos_quadrature_iterations(_value_context):
    """
    The maximum number of Lanczos iterations to perform when doing stochastic Lanczos
    quadrature. This is ONLY used for log determinant calculations and computing Tr(K^{-1}dK/d\theta)
    """
    _global_value = 15


class num_likelihood_samples(_value_context):
    """
    The number of samples to draw from a latent GP when computing a likelihood
    This is used in variational inference and training
    Default: 10
    """
    _global_value = 10


class num_trace_samples(_value_context):
    """
    The number of samples to draw when stochastically computing the trace of a matrix
    More values results in more accurate trace estimations
    If the value is set to 0, then the trace will be deterministically computed
    Default: 10
    """
    _global_value = 10


class use_toeplitz(_feature_flag):
    """
    Whether or not to use Toeplitz math with gridded data, grid inducing point modules
    Pros: memory efficient, faster on CPU
    Cons: slower on GPUs with < 10000 inducing points
    """
    _state = True
