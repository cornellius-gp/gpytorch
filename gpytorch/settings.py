#!/usr/bin/env python3


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

    def __enter__(self,):
        self.__class__._set_value(self._instance_value)

    def __exit__(self, *args):
        self.__class__._set_value(self._orig_value)
        return False


class check_training_data(_feature_flag):
    """
    Check whether the correct training data is supplied in Exact GP training mode
    Pros: fewer data checks, fewer warning messages
    Cons: possibility of supplying incorrect data, model accidentially in wrong mode

    Note: If using a Heteroskedastic Noise model, this will need to be disabled
    """

    _state = True


class debug(_feature_flag):
    """
    Whether or not to perform "safety" checks on the supplied data.
    (For example, that the correct training data is supplied in Exact GP training mode)
    Pros: fewer data checks, fewer warning messages
    Cons: possibility of supplying incorrect data, model accidentially in wrong mode
    """

    _state = True


class max_cg_iterations(_value_context):
    """
    The maximum number of conjugate gradient iterations to perform (when computing
    matrix solves). More values results in more accurate solves
    Default: 20
    """

    _global_value = 20


class max_cholesky_numel(_value_context):
    """
    If the number of elements of a LazyTensor is less than `max_cholesky_numel`,
    then the `root_decomposition` of LazyTensor will use Cholesky rather than Lanczos.
    Default: 256
    """

    _global_value = 256


class max_root_decomposition_size(_value_context):
    """
    The maximum number of Lanczos iterations to perform
    This is used when 1) computing variance estiamtes 2) when drawing from MVNs,
    or 3) for kernel multiplication
    More values results in higher accuracy
    Default: 100
    """

    _global_value = 100


class max_preconditioner_size(_value_context):
    """
    The maximum size of preconditioner to use. 0 corresponds to turning
    preconditioning off. When enabled, usually a value of around ~10 works fairly well.
    Default: 0
    """

    _global_value = 5


class max_lanczos_quadrature_iterations(_value_context):
    """
    The maximum number of Lanczos iterations to perform when doing stochastic
    Lanczos quadrature. This is ONLY used for log determinant calculations and
    computing Tr(K^{-1}dK/d\theta)
    """

    _global_value = 15


class memory_efficient(_feature_flag):
    """
    Whether or not to use Toeplitz math with gridded data, grid inducing point modules
    Pros: memory efficient, faster on CPU
    Cons: slower on GPUs with < 10000 inducing points
    """

    _state = False


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


class skip_logdet_forward(_feature_flag):
    """
    .. warning:

        ADVANCED FEATURE. Use this feature ONLY IF you're using
        `gpytorch.mlls.MarginalLogLikelihood` as loss functions for optimizing
        hyperparameters/variational parameters.  DO NOT use this feature if you
        need accurate estimates of the MLL (i.e. for model selection, MCMC,
        second order optimizaiton methods, etc.)

    This feature does not affect the gradients returned by
    :meth:`gpytorch.distributions.MultivariateNormal.log_prob`
    (used by `gpytorch.mlls.MarginalLogLikelihood`).
    The gradients remain unbiased estimates, and therefore can be used with SGD.
    However, the actual likelihood value returned by the forward
    pass will skip certain computations (i.e. the logdet computation), and will therefore
    be improper estimates.

    If you're using SGD (or a varient) to optimize parameters, you probably
    don't need an accurate MLL estimate; you only need accurate gradients. So
    this setting may give your model a performance boost.
    """

    _state = False


class terminate_cg_by_size(_feature_flag):
    """
    If set to true, cg will terminate after n iterations for an n x n matrix.
    """

    _state = True


class use_toeplitz(_feature_flag):
    """
    Whether or not to use Toeplitz math with gridded data, grid inducing point modules
    Pros: memory efficient, faster on CPU
    Cons: slower on GPUs with < 10000 inducing points
    """

    _state = True
