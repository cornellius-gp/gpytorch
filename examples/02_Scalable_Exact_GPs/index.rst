Exact GPs with Scalable (GPU) Inference
========================================

In GPyTorch, Exact GP inference is still our preferred approach to large regression datasets.
By coupling GPU acceleration with `BlackBox Matrix-Matrix Inference`_ and `LancZos Variance Estimates (LOVE)`_,
GPyTorch can perform inference on datasets with over 1,000,000 data points while making very few approximations.

How GPyTorch Scales Exact GPs
--------------------------------

GPyTorch relies on two key techniques to scale exact GPs to millions of data points using GPU acceleration.

- `BlackBox Matrix-Matrix Inference`_ (introduced by Gardner et al., 2018) computes the GP marginal log likelihood using only matrix multiplication.
  It is stochastic, but can scale exact GPs to millions of data points.
- `GP Regression (CUDA) with Fast Variances (LOVE)`_ demonstrates `LanczOs Variance Estimates (LOVE)`_ , a technique to rapidly speed up predictive variance computations.
  Check out this notebook to see how to use LOVE in GPyTorch, and how it compares to standard variance computations.

.. toctree::
   :maxdepth: 1
   :hidden:

   Simple_GP_Regression_With_LOVE_Fast_Variances_CUDA.ipynb

Exact GPs with GPU Acceleration
-----------------------------------

Here are examples of Exact GPs using GPU acceleration.

- For datasets with up to 10,000 data points, see our `single GPU regression example`_.
- For datasets with up to 1,000,000 data points, see our `multi GPU regression example`_.
- GPyTorch also integrates with KeOPs for extremely fast and memory-efficient kernel computations.
  See the `KeOPs integration notebook`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   Simple_GP_Regression_CUDA.ipynb
   Simple_MultiGPU_GP_Regression.ipynb
   KeOps_GP_Regression.ipynb

Scalable Kernel Approximations
-----------------------------------

While exact computations are our preferred approach, GPyTorch offer approximate kernels to reduce the asymptotic complexity of inference.

- `Sparse Gaussian Process Regression (SGPR)`_ (proposed by Titsias, 2009) which approximates kernels using a set of inducing points.
  This is a general purpose approximation.
- `Structured Kernel Interpolation (SKI/KISS-GP)`_ (proposed by Wilson and Nickish, 2015) which interpolates inducing points on a regularly spaced grid.
  This is designed for low-dimensional data and stationary kernels.
- `Structured Kernel Interpolation for Products (SKIP)`_ (proposed by Gardner et al., 2018) which extends SKI to higher dimensions.

.. toctree::
   :maxdepth: 1
   :hidden:

   SGPR_Regression_CUDA.ipynb
   KISSGP_Regression.ipynb
   Scalable_Kernel_Interpolation_for_Products_CUDA.ipynb

Structure-Exploiting Kernels
-----------------------------------

If your data lies on a Euclidean grid, and your GP uses a stationary kernel, the computations can be sped up dramatically.
See the `Grid Regression`_ example for more info.

.. toctree::
   :maxdepth: 1
   :hidden:

   Grid_GP_Regression.ipynb


.. _BlackBox Matrix-Matrix Inference:
  https://arxiv.org/abs/1809.11165

.. _GP Regression (CUDA) with Fast Variances (LOVE):
  ./Simple_GP_Regression_With_LOVE_Fast_Variances_CUDA.ipynb

.. _LancZos Variance Estimates (LOVE):
  https://arxiv.org/pdf/1803.06058.pdf

.. _single GPU regression example:
  ./Simple_GP_Regression_CUDA.ipynb

.. _multi GPU regression example:
  ./Simple_MultiGPU_GP_Regression.ipynb

.. _KeOPs integration notebook:
  ./KeOps_GP_Regression.ipynb

.. _Sparse Gaussian Process Regression (SGPR):
  ./SGPR_Regression_CUDA.ipynb

.. _Structured Kernel Interpolation (SKI/KISS-GP):
  ./KISSGP_Regression.ipynb

.. _Structured Kernel Interpolation for Products (SKIP):
  ./Scalable_Kernel_Interpolation_for_Products_CUDA.ipynb

.. _Grid Regression:
  ./Grid_GP_Regression.ipynb
