#!/usr/bin/env python3

import torch
from linear_operator.settings import (
    _linalg_dtype_cholesky,
    _linalg_dtype_symeig,
    cg_tolerance,
    cholesky_jitter,
    cholesky_max_tries,
    ciq_samples,
    deterministic_probes,
    fast_computations,
    linalg_dtypes,
    max_cg_iterations,
    max_cholesky_size,
    max_lanczos_quadrature_iterations,
    max_preconditioner_size,
    max_root_decomposition_size,
    min_preconditioning_size,
    minres_tolerance,
    num_contour_quadrature,
    num_trace_samples,
    preconditioner_tolerance,
    skip_logdet_forward,
    terminate_cg_by_size,
    tridiagonal_jitter,
    use_toeplitz,
    verbose_linalg,
)


class _dtype_value_context:
    _global_float_value = None
    _global_double_value = None
    _global_half_value = None

    @classmethod
    def value(cls, dtype):
        if torch.is_tensor(dtype):
            dtype = dtype.dtype
        if dtype == torch.float:
            return cls._global_float_value
        elif dtype == torch.double:
            return cls._global_double_value
        elif dtype == torch.half:
            return cls._global_half_value
        else:
            raise RuntimeError(f"Unsupported dtype for {cls.__name__}.")

    @classmethod
    def _set_value(cls, float_value, double_value, half_value):
        if float_value is not None:
            cls._global_float_value = float_value
        if double_value is not None:
            cls._global_double_value = double_value
        if half_value is not None:
            cls._global_half_value = half_value

    def __init__(self, float=None, double=None, half=None):
        self._orig_float_value = self.__class__.value()
        self._instance_float_value = float
        self._orig_double_value = self.__class__.value()
        self._instance_double_value = double
        self._orig_half_value = self.__class__.value()
        self._instance_half_value = half

    def __enter__(
        self,
    ):
        self.__class__._set_value(
            self._instance_float_value,
            self._instance_double_value,
            self._instance_half_value,
        )

    def __exit__(self, *args):
        self.__class__._set_value(self._orig_float_value, self._orig_double_value, self._orig_half_value)
        return False


class _feature_flag:
    r"""Base class for feature flag settings with global scope.
    The default is set via the `_default` class attribute.
    """

    _default = False
    _state = None

    @classmethod
    def is_default(cls):
        return cls._state is None

    @classmethod
    def on(cls):
        if cls.is_default():
            return cls._default
        return cls._state

    @classmethod
    def off(cls):
        return not cls.on()

    @classmethod
    def _set_state(cls, state):
        cls._state = state

    def __init__(self, state=True):
        self.prev = self.__class__._state
        self.state = state

    def __enter__(self):
        self.__class__._set_state(self.state)

    def __exit__(self, *args):
        self.__class__._set_state(self.prev)
        return False


class _value_context:
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

    def __enter__(
        self,
    ):
        self.__class__._set_value(self._instance_value)

    def __exit__(self, *args):
        self.__class__._set_value(self._orig_value)
        return False


class debug(_feature_flag):
    """
    Whether or not to perform "safety" checks on the supplied data.
    (For example, that the correct training data is supplied in Exact GP training mode)
    Pros: fewer data checks, fewer warning messages
    Cons: possibility of supplying incorrect data, model accidentially in wrong mode

    (Default: True)
    """

    _default = True


class detach_test_caches(_feature_flag):
    """
    Whether or not to detach caches computed for making predictions. In most cases, you will want this,
    as this will speed up derivative computations of the predictions with respect to test inputs. However,
    if you also need derivatives with respect to training inputs (e.g., because you have fantasy observations),
    then you must disable this.

    (Default: True)
    """

    _default = True


class eval_cg_tolerance(_value_context):
    """
    Relative residual tolerance to use for terminating CG when making predictions.

    (Default: 1e-2)
    """

    _global_value = 0.01


class fast_pred_var(_feature_flag):
    """
    Fast predictive variances using Lanczos Variance Estimates (LOVE)
    Use this for improved performance when computing predictive variances.

    As described in the paper:

    `Constant-Time Predictive Distributions for Gaussian Processes`_.

    See also: :class:`gpytorch.settings.max_root_decomposition_size` (to control the
    size of the low rank decomposition used for variance estimates).

    (Default: False)

    .. _`Constant-Time Predictive Distributions for Gaussian Processes`:
        https://arxiv.org/pdf/1803.06058.pdf
    """

    _num_probe_vectors = 1

    @classmethod
    def num_probe_vectors(cls):
        return cls._num_probe_vectors

    @classmethod
    def _set_num_probe_vectors(cls, value):
        cls._num_probe_vectors = value

    def __init__(self, state=True, num_probe_vectors=1):
        self.orig_value = self.__class__.num_probe_vectors()
        self.value = num_probe_vectors
        super().__init__(state)

    def __enter__(self):
        self.__class__._set_num_probe_vectors(self.value)
        super().__enter__()

    def __exit__(self, *args):
        self.__class__._set_num_probe_vectors(self.orig_value)
        return super().__exit__()


class fast_pred_samples(_feature_flag):
    """
    Fast predictive samples using Lanczos Variance Estimates (LOVE).
    Use this for improved performance when sampling from a predictive posterior matrix.

    As described in the paper:

    `Constant-Time Predictive Distributions for Gaussian Processes`_.

    See also: :class:`gpytorch.settings.max_root_decomposition_size` (to control the
    size of the low rank decomposition used for samples).

    (Default: False)

    .. _`Constant-Time Predictive Distributions for Gaussian Processes`:
        https://arxiv.org/pdf/1803.06058.pdf
    """

    _default = False


class lazily_evaluate_kernels(_feature_flag):
    """
    Lazily compute the entries of covariance matrices (set to True by default).
    This can result in memory and speed savings - if say cross covariance terms are not needed
    or if you only need to compute variances (not covariances).

    If set to False, gpytorch will always compute the entire covariance matrix between
    training and test data.

    (Default: True)
    """

    _default = True


class max_eager_kernel_size(_value_context):
    """
    If the joint train/test covariance matrix is less than this size, then we will avoid as
    much lazy evaluation of the kernel as possible.

    (Default: 512)
    """

    _global_value = 512


class memory_efficient(_feature_flag):
    """
    Whether or not to use Toeplitz math with gridded data, grid inducing point modules
    Pros: memory efficient, faster on CPU
    Cons: slower on GPUs with < 10000 inducing points

    (Default: False)
    """

    _default = False


class min_fixed_noise(_dtype_value_context):
    """
    The minimum noise value that can be used in :obj:`~gpytorch.likelihoods.FixedNoiseGaussianLikelihood`.
    If the supplied noise values are smaller than this, they are rounded up and a warning is raised.

    - Default for `float`: 1e-4
    - Default for `double`: 1e-6
    - Default for `half`: 1e-3
    """

    _global_float_value = 1e-4
    _global_double_value = 1e-6
    _global_half_value = 1e-3


class min_variance(_dtype_value_context):
    """
    The minimum variance that can be returned from :obj:`~gpytorch.distributions.MultivariateNormal#variance`.
    If variances are smaller than this, they are rounded up and a warning is raised.

    - Default for `float`: 1e-6
    - Default for `double`: 1e-10
    - Default for `half`: 1e-3
    """

    _global_float_value = 1e-6
    _global_double_value = 1e-10
    _global_half_value = 1e-3


class num_gauss_hermite_locs(_value_context):
    """
    The number of samples to draw from a latent GP when computing a likelihood
    This is used in variational inference and training

    (Default: 20)
    """

    _global_value = 20


class num_likelihood_samples(_value_context):
    """
    The number of samples to draw from a latent GP when computing a likelihood
    This is used in variational inference and training

    (Default: 10)
    """

    _global_value = 10


class prior_mode(_feature_flag):
    """
    If set to true, GP models will be evaluated in prior mode.
    This allows evaluating any Exact GP model in prior mode, even it if has training data / targets.

    (Default: False)
    """

    _default = False


class sgpr_diagonal_correction(_feature_flag):
    """
    If set to true, during posterior prediction the variances of the InducingPointKernel
    will be corrected to match the variances of the exact kernel.

    If false then no such correction will be performed (this is the default in other libraries).

    (Default: True)
    """

    _default = True


class skip_posterior_variances(_feature_flag):
    """
    Whether or not to skip the posterior covariance matrix when doing an ExactGP
    forward pass. If this is on, the returned gpytorch MultivariateNormal will have a
    ZeroLinearOperator as its covariance matrix. This allows gpytorch to not compute
    the covariance matrix when it is not needed, speeding up computations.

    (Default: False)
    """

    _default = False


class trace_mode(_feature_flag):
    """
    If set to True, we will generally try to avoid calling our built in PyTorch functions, because these cannot
    be run through torch.jit.trace.

    Note that this will sometimes involve explicitly evaluating lazy tensors and various other slowdowns and
    inefficiencies. As a result, you really shouldn't use this feature context unless you are calling torch.jit.trace
    on a GPyTorch model.

    Our hope is that this flag will not be necessary long term, once https://github.com/pytorch/pytorch/issues/22329
    is fixed.

    (Default: False)
    """

    _default = False


__all__ = [
    "_linalg_dtype_symeig",
    "_linalg_dtype_cholesky",
    "cg_tolerance",
    "cholesky_jitter",
    "cholesky_max_tries",
    "ciq_samples",
    "debug",
    "detach_test_caches",
    "deterministic_probes",
    "eval_cg_tolerance",
    "fast_computations",
    "fast_pred_var",
    "fast_pred_samples",
    "lazily_evaluate_kernels",
    "linalg_dtypes",
    "max_eager_kernel_size",
    "max_cholesky_size",
    "max_cg_iterations",
    "max_lanczos_quadrature_iterations",
    "max_preconditioner_size",
    "max_root_decomposition_size",
    "memory_efficient",
    "min_preconditioning_size",
    "min_variance",
    "minres_tolerance",
    "num_contour_quadrature",
    "num_gauss_hermite_locs",
    "num_likelihood_samples",
    "num_trace_samples",
    "preconditioner_tolerance",
    "prior_mode",
    "sgpr_diagonal_correction",
    "skip_logdet_forward",
    "skip_posterior_variances",
    "terminate_cg_by_size",
    "trace_mode",
    "tridiagonal_jitter",
    "use_toeplitz",
    "verbose_linalg",
]
