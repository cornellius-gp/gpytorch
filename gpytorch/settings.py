#!/usr/bin/env python3

import logging
import warnings

import torch


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

    def __enter__(self,):
        self.__class__._set_value(self._instance_value)

    def __exit__(self, *args):
        self.__class__._set_value(self._orig_value)
        return False


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

    def __enter__(self,):
        self.__class__._set_value(
            self._instance_float_value, self._instance_double_value, self._instance_half_value,
        )

    def __exit__(self, *args):
        self.__class__._set_value(self._orig_float_value, self._orig_double_value, self._orig_half_value)
        return False


class _fast_covar_root_decomposition(_feature_flag):
    r"""
    This feature flag controls how matrix root decompositions (:math:`K = L L^\top`) are computed
    (e.g. for sampling, computing caches, etc.).

    If set to True, covariance matrices :math:`K` are decomposed with low-rank approximations :math:`L L^\top`,
    (:math:`L \in \mathbb R^{n \times k}`) using the Lanczos algorithm.
    This is faster for large matrices and exploits structure in the covariance matrix if applicable.

    If set to False, covariance matrices :math:`K` are decomposed using the Cholesky decomposition.

    .. warning ::

        Setting this to False will compute a complete Cholesky decomposition of covariance matrices.
        This may be infeasible for GPs with structure covariance matrices.

    See also: :class:`gpytorch.settings.max_root_decomposition_size` (to control the
    size of the low rank decomposition used).
    """

    _default = True


class _fast_log_prob(_feature_flag):
    r"""
    This feature flag controls how to compute the marginal log likelihood of exact GPs
    and the log probability of multivariate normal distributions.

    If set to True, log_prob is computed using a modified conjugate gradients algorithm (as
    described in `GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration`_.
    This is a stochastic computation, but it is much faster for large matrices
    and exploits structure in the covariance matrix if applicable.

    If set to False, `log_prob` is computed using the Cholesky decomposition.

    .. warning ::

        Setting this to False will compute a complete Cholesky decomposition of covariance matrices.
        This may be infeasible for GPs with structure covariance matrices.

    See also: :class:`gpytorch.settings.num_trace_samples` (to control the
    stochasticity of the fast `log_prob` estimates).

    .. _GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration:
        https://arxiv.org/pdf/1809.11165.pdf
    """

    _default = True


class _fast_solves(_feature_flag):
    r"""
    This feature flag controls how to compute solves with positive definite matrices.
    If set to True, solves are computed using preconditioned conjugate gradients.
    If set to False, `log_prob` is computed using the Cholesky decomposition.

    .. warning ::

        Setting this to False will compute a complete Cholesky decomposition of covariance matrices.
        This may be infeasible for GPs with structure covariance matrices.
    """

    _default = True


class skip_posterior_variances(_feature_flag):
    """
    Whether or not to skip the posterior covariance matrix when doing an ExactGP
    forward pass. If this is on, the returned gpytorch MultivariateNormal will have a
    ZeroLazyTensor as its covariance matrix. This allows gpytorch to not compute
    the covariance matrix when it is not needed, speeding up computations.

    (Default: False)
    """

    _default = False


class detach_test_caches(_feature_flag):
    """
    Whether or not to detach caches computed for making predictions. In most cases, you will want this,
    as this will speed up derivative computations of the predictions with respect to test inputs. However,
    if you also need derivatives with respect to training inputs (e.g., because you have fantasy observations),
    then you must disable this.

    (Default: True)
    """

    _default = True


class deterministic_probes(_feature_flag):
    """
    Whether or not to resample probe vectors every iteration of training. If True, we use the same set of probe vectors
    for computing log determinants each iteration. This introduces small amounts of bias in to the MLL, but allows us
    to compute a deterministic estimate of it which makes optimizers like L-BFGS more viable choices.

    NOTE: Currently, probe vectors are cached in a global scope. Therefore, this setting cannot be used
    if multiple independent GP models are being trained in the same context (i.e., it works fine with a single GP model)

    (Default: False)
    """

    probe_vectors = None

    @classmethod
    def _set_state(cls, state):
        super()._set_state(state)
        cls.probe_vectors = None


class debug(_feature_flag):
    """
    Whether or not to perform "safety" checks on the supplied data.
    (For example, that the correct training data is supplied in Exact GP training mode)
    Pros: fewer data checks, fewer warning messages
    Cons: possibility of supplying incorrect data, model accidentially in wrong mode

    (Default: True)
    """

    _default = True


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


class fast_computations:
    r"""
    This feature flag controls whether or not to use fast approximations to various mathematical
    functions used in GP inference.
    The functions that can be controlled are:

    * :attr:`covar_root_decomposition`
        This feature flag controls how matrix root decompositions
        (:math:`K = L L^\top`) are computed (e.g. for sampling, computing caches, etc.).

        * If set to True,
            covariance matrices :math:`K` are decomposed with low-rank approximations :math:`L L^\top`,
            (:math:`L \in \mathbb R^{n \times k}`) using the Lanczos algorithm.
            This is faster for large matrices and exploits structure in the covariance matrix if applicable.

        * If set to False,
            covariance matrices :math:`K` are decomposed using the Cholesky decomposition.

    * :attr:`log_prob`
        This feature flag controls how GPyTorch computes the marginal log likelihood for exact GPs
        and `log_prob` for multivariate normal distributions

        * If set to True,
            `log_prob` is computed using a modified conjugate gradients algorithm (as
            described in `GPyTorch Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration`_.
            This is a stochastic computation, but it is much faster for large matrices
            and exploits structure in the covariance matrix if applicable.

        * If set to False,
            `log_prob` is computed using the Cholesky decomposition.

    * :attr:`fast_solves`
        This feature flag controls how GPyTorch computes the solves of positive-definite matrices.

        * If set to True,
            Solves are computed with preconditioned conjugate gradients.

        * If set to False,
            Solves are computed using the Cholesky decomposition.

    .. warning ::

        Setting this to False will compute a complete Cholesky decomposition of covariance matrices.
        This may be infeasible for GPs with structure covariance matrices.

    By default, approximations are used for all of these functions (except for solves).
    Setting any of them to False will use exact computations instead.

    See also:
        * :class:`gpytorch.settings.max_root_decomposition_size`
            (to control the size of the low rank decomposition used)
        * :class:`gpytorch.settings.num_trace_samples`
            (to control the stochasticity of the fast `log_prob` estimates)

    .. _GPyTorch Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration:
        https://arxiv.org/pdf/1809.11165.pdf
    """
    covar_root_decomposition = _fast_covar_root_decomposition
    log_prob = _fast_log_prob
    solves = _fast_solves

    def __init__(self, covar_root_decomposition=True, log_prob=True, solves=True):
        self.covar_root_decomposition = _fast_covar_root_decomposition(covar_root_decomposition)
        self.log_prob = _fast_log_prob(log_prob)
        self.solves = _fast_solves(solves)

    def __enter__(self):
        self.covar_root_decomposition.__enter__()
        self.log_prob.__enter__()
        self.solves.__enter__()

    def __exit__(self, *args):
        self.covar_root_decomposition.__exit__()
        self.log_prob.__exit__()
        self.solves.__exit__()
        return False


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


class max_cg_iterations(_value_context):
    """
    The maximum number of conjugate gradient iterations to perform (when computing
    matrix solves). A higher value rarely results in more accurate solves -- instead, lower the CG tolerance.

    (Default: 1000)
    """

    _global_value = 1000


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


class cholesky_jitter(_dtype_value_context):
    """
    The jitter value passed to `psd_safe_cholesky` when using cholesky solves.

    - Default for `float`: 1e-6
    - Default for `double`: 1e-8
    """

    _global_float_value = 1e-6
    _global_double_value = 1e-8

    @classmethod
    def value(cls, dtype=None):
        if dtype is None:
            # Deprecated in 1.4: remove in 1.5
            warnings.warn(
                "cholesky_jitter is now a _dtype_value_context and should be called with a dtype argument",
                DeprecationWarning,
            )
            return cls._global_float_value
        return super().value(dtype=dtype)


class cg_tolerance(_value_context):
    """
    Relative residual tolerance to use for terminating CG.

    (Default: 1)
    """

    _global_value = 1


class ciq_samples(_feature_flag):
    """
    Whether to draw samples using Contour Integral Quadrature or not.
    This may be slower than standard sampling methods for `N < 5000`.
    However, it should be faster with larger matrices.

    As described in the paper:

    `Fast Matrix Square Roots with Applications to Gaussian Processes and Bayesian Optimization`_.

    (Default: False)

    .. _`Fast Matrix Square Roots with Applications to Gaussian Processes and Bayesian Optimization`:
        https://arxiv.org/abs/2006.11267
    """

    _default = False


class preconditioner_tolerance(_value_context):
    """
    Diagonal trace tolerance to use for checking preconditioner convergence.

    (Default: 1e-3)
    """

    _global_value = 1e-3


class eval_cg_tolerance(_value_context):
    """
    Relative residual tolerance to use for terminating CG when making predictions.

    (Default: 1e-2)
    """

    _global_value = 0.01


class _use_eval_tolerance(_feature_flag):
    _default = False


class max_cholesky_size(_value_context):
    """
    If the size of of a LazyTensor is less than `max_cholesky_size`,
    then `root_decomposition` and `inv_matmul` of LazyTensor will use Cholesky rather than Lanczos/CG.

    (Default: 800)
    """

    _global_value = 800


class max_root_decomposition_size(_value_context):
    """
    The maximum number of Lanczos iterations to perform
    This is used when 1) computing variance estiamtes 2) when drawing from MVNs,
    or 3) for kernel multiplication
    More values results in higher accuracy

    (Default: 100)
    """

    _global_value = 100


class max_preconditioner_size(_value_context):
    """
    The maximum size of preconditioner to use. 0 corresponds to turning
    preconditioning off. When enabled, usually a value of around ~10 works fairly well.

    (Default: 15)
    """

    _global_value = 15


class max_lanczos_quadrature_iterations(_value_context):
    r"""
    The maximum number of Lanczos iterations to perform when doing stochastic
    Lanczos quadrature. This is ONLY used for log determinant calculations and
    computing Tr(K^{-1}dK/d\theta)

    (Default: 20)
    """

    _global_value = 20


class memory_efficient(_feature_flag):
    """
    Whether or not to use Toeplitz math with gridded data, grid inducing point modules
    Pros: memory efficient, faster on CPU
    Cons: slower on GPUs with < 10000 inducing points

    (Default: False)
    """

    _default = False


class min_preconditioning_size(_value_context):
    """
    If the size of of a LazyTensor is less than `min_preconditioning_size`,
    then we won't use pivoted Cholesky based preconditioning.

    (Default: 2000)
    """

    _global_value = 2000


class minres_tolerance(_value_context):
    """
    Relative update term tolerance to use for terminating MINRES.

    (Default: 1e-4)
    """

    _global_value = 1e-4


class num_contour_quadrature(_value_context):
    """
    The number of quadrature points to compute CIQ.

    (Default: 15)
    """

    _global_value = 15


class num_likelihood_samples(_value_context):
    """
    The number of samples to draw from a latent GP when computing a likelihood
    This is used in variational inference and training

    (Default: 10)
    """

    _global_value = 10


class num_gauss_hermite_locs(_value_context):
    """
    The number of samples to draw from a latent GP when computing a likelihood
    This is used in variational inference and training

    (Default: 20)
    """

    _global_value = 20


class num_trace_samples(_value_context):
    """
    The number of samples to draw when stochastically computing the trace of a matrix
    More values results in more accurate trace estimations
    If the value is set to 0, then the trace will be deterministically computed

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

    (Default: False)
    """

    _default = False


class terminate_cg_by_size(_feature_flag):
    """
    If set to true, cg will terminate after n iterations for an n x n matrix.

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


class tridiagonal_jitter(_value_context):
    """
    The (relative) amount of noise to add to the diagonal of tridiagonal matrices before
    eigendecomposing. root_decomposition becomes slightly more stable with this, as we need
    to take the square root of the eigenvalues. Any eigenvalues still negative after adding jitter
    will be zeroed out.

    (Default: 1e-6)
    """

    _global_value = 1e-6


class use_toeplitz(_feature_flag):
    """
    Whether or not to use Toeplitz math with gridded data, grid inducing point modules
    Pros: memory efficient, faster on CPU
    Cons: slower on GPUs with < 10000 inducing points

    (Default: True)
    """

    _default = True


class verbose_linalg(_feature_flag):
    """
    Print out information whenever running an expensive linear algebra routine (e.g. Cholesky, CG, Lanczos, CIQ, etc.)

    (Default: False)
    """

    _default = False

    # Create a global logger
    logger = logging.getLogger("LinAlg (Verbose)")
    logger.setLevel(logging.DEBUG)

    # Output logging results to the stdout stream
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
