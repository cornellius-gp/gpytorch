#!/usr/bin/env python3


class _feature_flag(object):
    _state = False

    @classmethod
    def on(cls):
        return cls._state

    @classmethod
    def off(cls):
        return (not cls._state)

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

    _state = True


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

    _state = True


class check_training_data(_feature_flag):
    """
    Check whether the correct training data is supplied in Exact GP training mode
    Pros: fewer data checks, fewer warning messages
    Cons: possibility of supplying incorrect data, model accidentially in wrong mode

    Note: If using a Heteroskedastic Noise model, this will need to be disabled
    """

    _state = True


class skip_posterior_variances(_feature_flag):
    """
    Whether or not to skip the posterior covariance matrix when doing an ExactGP
    forward pass. If this is on, the returned gpytorch MultivariateNormal will have a
    ZeroLazyTensor as its covariance matrix. This allows gpytorch to not compute
    the covariance matrix when it is not needed, speeding up computations.
    """
    _state = False


class detach_test_caches(_feature_flag):
    """
    Whether or not to detach caches computed for making predictions. In most cases, you will want this,
    as this will speed up derivative computations of the predictions with respect to test inputs. However,
    if you also need derivatives with respect to training inputs (e.g., because you have fantasy observations),
    then you must disable this.
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


class fast_pred_var(_feature_flag):
    """
    Fast predictive variances using Lanczos Variance Estimates (LOVE)
    Use this for improved performance when computing predictive variances.

    As described in the paper:

    `Constant-Time Predictive Distributions for Gaussian Processes`_.

    See also: :class:`gpytorch.settings.max_root_decomposition_size` (to control the
    size of the low rank decomposition used for variance estimates).

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
        super(fast_pred_var, self).__init__(state)

    def __enter__(self):
        self.__class__._set_num_probe_vectors(self.value)
        super(fast_pred_var, self).__enter__()

    def __exit__(self, *args):
        self.__class__._set_num_probe_vectors(self.orig_value)
        return super(fast_pred_var, self).__exit__()


class fast_pred_samples(_feature_flag):
    """
    Fast predictive samples using Lanczos Variance Estimates (LOVE).
    Use this for improved performance when sampling from a predictive posterior matrix.

    As described in the paper:

    `Constant-Time Predictive Distributions for Gaussian Processes`_.

    See also: :class:`gpytorch.settings.max_root_decomposition_size` (to control the
    size of the low rank decomposition used for samples).

    .. _`Constant-Time Predictive Distributions for Gaussian Processes`:
        https://arxiv.org/pdf/1803.06058.pdf
    """

    pass


class fast_computations(object):
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

    .. warning ::

        Setting this to False will compute a complete Cholesky decomposition of covariance matrices.
        This may be infeasible for GPs with structure covariance matrices.

    By default, approximations are used for all of these functions. Setting any of them to False will use
    exact computations instead.

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

    def __init__(self, covar_root_decomposition=True, log_prob=True):
        self.covar_root_decomposition = _fast_covar_root_decomposition(covar_root_decomposition)
        self.log_prob = _fast_log_prob(log_prob)

    def __enter__(self):
        self.covar_root_decomposition.__enter__()
        self.log_prob.__enter__()

    def __exit__(self, *args):
        self.covar_root_decomposition.__exit__()
        self.log_prob.__exit__()
        return False


class lazily_evaluate_kernels(_feature_flag):
    """
    Lazily compute the entries of covariance matrices (set to True by default).
    This can result in memory and speed savings - if say cross covariance terms are not needed
    or if you only need to compute variances (not covariances).

    If set to False, gpytorch will always compute the entire covariance matrix between
    training and test data.
    """

    _state = True


class max_cg_iterations(_value_context):
    """
    The maximum number of conjugate gradient iterations to perform (when computing
    matrix solves). A higher value rarely results in more accurate solves -- instead, lower the CG tolerance.
    Default: 1000
    """

    _global_value = 1000


class cg_tolerance(_value_context):
    """
    Relative residual tolerance to use for terminating CG.

    Default: 1
    """

    _global_value = 1


class preconditioner_tolerance(_value_context):
    """
    Diagonal trace tolerance to use for checking preconditioner convergence.

    Default: 1e-3
    """

    _global_value = 1e-3


class eval_cg_tolerance(_value_context):
    """
    Relative residual tolerance to use for terminating CG when making predictions.

    Default: 0.01
    """

    _global_value = 0.01


class _use_eval_tolerance(_feature_flag):
    _state = False


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

    _global_value = 15


class max_lanczos_quadrature_iterations(_value_context):
    """
    The maximum number of Lanczos iterations to perform when doing stochastic
    Lanczos quadrature. This is ONLY used for log determinant calculations and
    computing Tr(K^{-1}dK/d\theta)
    """

    _global_value = 20


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


class num_gauss_hermite_locs(_value_context):
    """
    The number of samples to draw from a latent GP when computing a likelihood
    This is used in variational inference and training
    Default: 10
    """

    _global_value = 20


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

    _state = False


class tridiagonal_jitter(_value_context):
    """
    The (relative) amount of noise to add to the diagonal of tridiagonal matrices before
    eigendecomposing. root_decomposition becomes slightly more stable with this, as we need
    to take the square root of the eigenvalues. Any eigenvalues still negative after adding jitter
    will be zeroed out.
    """

    _global_value = 1e-6


class use_toeplitz(_feature_flag):
    """
    Whether or not to use Toeplitz math with gridded data, grid inducing point modules
    Pros: memory efficient, faster on CPU
    Cons: slower on GPUs with < 10000 inducing points
    """

    _state = True
