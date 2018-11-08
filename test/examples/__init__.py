from .test_batch_gp_regression import TestBatchGPRegression
from .test_batch_multitask_gp_regression import TestBatchMultitaskGPRegression
from .test_hadamard_multitask_gp_regression import TestHadamardMultitaskGPRegression
from .test_kissgp_additive_classification import TestKISSGPAdditiveClassification
from .test_kissgp_additive_regression import TestKISSGPAdditiveRegression
from .test_kissgp_dkl_regression import TestDKLRegression
from .test_kissgp_gp_classification import TestKISSGPClassification
from .test_kissgp_gp_regression import TestKISSGPRegression
from .test_kissgp_kronecker_product_classification import TestKISSGPKroneckerProductClassification
from .test_kissgp_kronecker_product_regression import TestKISSGPKroneckerProductRegression
from .test_kissgp_multiplicative_regression import TestKISSGPMultiplicativeRegression
from .test_kissgp_variational_regression import TestKISSGPVariationalRegression
from .test_kissgp_white_noise_regression import TestKISSGPWhiteNoiseRegression
from .test_kronecker_multitask_gp_regression import TestKroneckerMultiTaskGPRegression
from .test_kronecker_multitask_ski_gp_regression import TestKroneckerMultiTaskKISSGPRegression
from .test_lcm_kernel_regression import TestLCMKernelRegression
from .test_sgpr_regression import TestSGPRRegression
from .test_simple_gp_classification import TestSimpleGPClassification
from .test_simple_gp_regression import TestSimpleGPRegression
from .test_spectral_mixture_gp_regression import TestSpectralMixtureGPRegression
from .test_svgp_gp_regression import TestSVGPRegression
from .test_white_noise_regression import TestWhiteNoiseGPRegression

__all__ = [
    "TestBatchGPRegression",
    "TestBatchMultitaskGPRegression",
    "TestHadamardMultitaskGPRegression",
    "TestKISSGPAdditiveClassification",
    "TestKISSGPAdditiveRegression",
    "TestDKLRegression",
    "TestKISSGPClassification",
    "TestKISSGPRegression",
    "TestKISSGPKroneckerProductClassification",
    "TestKISSGPKroneckerProductRegression",
    "TestKISSGPMultiplicativeRegression",
    "TestKISSGPVariationalRegression",
    "TestKISSGPWhiteNoiseRegression",
    "TestKroneckerMultiTaskGPRegression",
    "TestKroneckerMultiTaskKISSGPRegression",
    "TestLCMKernelRegression",
    "TestSGPRRegression",
    "TestSimpleGPClassification",
    "TestSimpleGPRegression",
    "TestSpectralMixtureGPRegression",
    "TestSVGPRegression",
    "TestWhiteNoiseGPRegression",
]
